FROM pytorch/pytorch:latest
LABEL authors="netRunner57"

RUN apt-get update && apt-get install -y git

RUN pip install transformers accelerate evaluate bitsandbytes peft einops safetensors xformers langchain ctransformers[cuda] chromadb sentence-transformers

RUN pip install langserve[all]

RUN pip install -U langchain-cli

CMD ["sh", "-c", "while true; do sleep 3600; done"]