FROM 763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2 

RUN mkdir /workspace

WORKDIR /workspace

COPY scripts /workspace

RUN pip install /workspace/requirements.txt

RUN pip install hyperpod-elastic-agent

