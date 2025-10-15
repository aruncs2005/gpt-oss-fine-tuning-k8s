#!/usr/bin/env python3
"""
Multinode QAT Fine-tuning for GPT-OSS on EKS
Adapted for 16 GPU P5 cluster with full fine-tuning
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Mxfp4Config
from trl import SFTTrainer, SFTConfig, ModelConfig, ScriptArguments
from datasets import load_dataset
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import logging
import argparse
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_distributed():
    """Initialize distributed training environment"""
    # Get environment variables set by torchrun or Kubernetes
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    logger.info(f"Initialized distributed training: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    return rank, local_rank, world_size, device

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def load_model_and_tokenizer(model_name: str, local_rank: int):
    """Load model and tokenizer with proper device mapping for distributed training"""
    
    # Model configuration
    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
        "use_cache": False,
        # Don't use device_map="auto" in distributed training
        "device_map": None,
    }
    
    # Handle MXFP4 dequantization if needed
    config = AutoConfig.from_pretrained(model_name)
    if (
        getattr(config, "quantization_config", {})
        and config.quantization_config.get("quant_method", None) == "mxfp4"
    ):
        model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move model to GPU
    model = model.cuda(local_rank)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_datasets(dataset_name: str, test_size: float = 0.1):
    """Load and prepare training datasets"""
    dataset = load_dataset(dataset_name)
    dataset = dataset["train"].train_test_split(test_size=test_size, seed=42)
    
    return dataset["train"], dataset["test"]

def create_training_config(output_dir: str, world_size: int, local_rank: int):
    """Create distributed training configuration with DeepSpeed ZeRO-3"""
    
    # P5.48xlarge has 8x H100 GPUs with 80GB each
    # With ZeRO-3, we can use larger batch sizes due to memory savings
    per_device_batch_size = 1  # Increased thanks to ZeRO-3 memory efficiency
    gradient_accumulation_steps = 2  # Effective batch size = 8 * 2 * 16 = 256
    
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3.0,
        learning_rate=2e-5,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=4096,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        
        # DeepSpeed ZeRO-3 configuration
       # deepspeed="ds_config.json",
        

        #FSDP
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "GptOssDecoderLayer", # update as per your model
        },
        # Distributed training settings
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        
        # Evaluation and logging
        eval_strategy="steps",
        eval_on_start=False,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Optimization settings (DeepSpeed will override some of these)
        weight_decay=0.01,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        
        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        bf16_full_eval=True,
        
        # Reporting
        report_to="tensorboard" if local_rank == 0 else None,
        logging_dir=f"{output_dir}/logs" if local_rank == 0 else None,
        logging_first_step=True,
    )

def apply_quantization(model, trainer, calib_size: int = 128):
    """Apply quantization-aware training setup"""
    
    # Use MXFP4 configuration for weight-only quantization
    quantization_config = mtq.MXFP4_MLP_WEIGHT_ONLY_CFG
    
    # Prepare calibration dataset
    calib_dataset = torch.utils.data.Subset(
        trainer.eval_dataset, 
        list(range(min(len(trainer.eval_dataset), calib_size)))
    )
    data_loader = trainer.get_eval_dataloader(calib_dataset)
    
    def forward_loop(model):
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                if i >= calib_size // trainer.args.per_device_eval_batch_size:
                    break
                # Move data to correct device
                data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                model(**data)
    
    # Apply quantization
    logger.info("Applying quantization configuration...")
    mtq.quantize(model, quantization_config, forward_loop)
    logger.info("Quantization applied successfully")


def main():
    parser = argparse.ArgumentParser(description="Multinode QAT Fine-tuning with DeepSpeed ZeRO-3")
    parser.add_argument("--model-name", default="openai/gpt-oss-20b", help="Model name or path")
    parser.add_argument("--dataset-name", default="HuggingFaceH4/Multilingual-Thinking", help="Dataset name")
    parser.add_argument("--output-dir", default="/shared/output", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test split size")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DeepSpeed")
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, local_rank, world_size, device = setup_distributed()
    
    try:
        # Enable modelopt checkpointing
        mto.enable_huggingface_checkpointing()
        
        if rank == 0:
            logger.info(f"Starting multinode QAT fine-tuning with DeepSpeed ZeRO-3")
            logger.info(f"GPUs: {world_size}")
            logger.info(f"Model: {args.model_name}")
            logger.info(f"Dataset: {args.dataset_name}")
            logger.info(f"Output: {args.output_dir}")
            
        
        # Wait for rank 0 to create config
        dist.barrier()
        
        # Load model and tokenizer
        logger.info(f"Loading model on rank {rank}")
        model, tokenizer = load_model_and_tokenizer(args.model_name, local_rank)
        
        # Prepare datasets
        if rank == 0:
            logger.info("Loading datasets...")
        train_dataset, eval_dataset = prepare_datasets(args.dataset_name, args.test_size)
        
        # Create training configuration
        training_args = create_training_config(args.output_dir, world_size, local_rank)
        
       # Apply quantization BEFORE trainer initialization
        # This ensures all ranks have identical model structure for DeepSpeed ZeRO-3
        # if rank == 0:
        #     logger.info("Applying quantization before trainer initialization...")
        
        # # Use a simple forward loop for calibration
        # quantization_config = mtq.MXFP4_MLP_WEIGHT_ONLY_CFG
        
        # def forward_loop(model):
        #     # Minimal calibration - just initialize quantization parameters
        #     # We'll do proper calibration after trainer is set up if needed
        #     pass
        
        # logger.info(f"Rank {rank}: Applying quantization...")
        # mtq.quantize(model, quantization_config, forward_loop)
        
        # Critical: Synchronize to ensure all ranks have identical model structure
        dist.barrier()
        if rank == 0:
            logger.info("Quantization applied consistently across all ranks")

        # Initialize trainer
        logger.info(f"Initializing trainer on rank {rank}")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        
        # Apply quantization (before DeepSpeed initialization)
        if rank == 0:
            logger.info("Setting up quantization...")
        apply_quantization(model, trainer)
        
        # DeepSpeed will handle model wrapping automatically
        # ZeRO-3 shards model parameters across all GPUs
        
        # Start training
        if rank == 0:
            logger.info("Starting training...")
        
        trainer.train()
        
        # Save final model (only on main process)
        if rank == 0:
            logger.info("Saving final model...")
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logger.info(f"Model saved to {args.output_dir}")
        
        # Wait for all processes to complete
        dist.barrier()
        
        if rank == 0:
            logger.info("Training completed successfully!")
            
    except Exception as e:
        logger.error(f"Training failed on rank {rank}: {str(e)}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()