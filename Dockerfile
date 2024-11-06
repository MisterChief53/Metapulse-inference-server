# Use the official PyTorch image with CUDA 12.1
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
LABEL authors="netRunner57"

# Install system dependencies
RUN apt-get update && apt-get install -y git curl wget build-essential

# RUN pip install bitsandbytes

# # Install Hugging Face libraries and others
RUN pip install transformers[torch] accelerate evaluate peft xformers sentence-transformers chromadb langchain langchain-community langserve[all]

# # Install langchain-cli if needed
RUN pip install -U langchain-cli

# Set environment variables for CUDA (optional)
# ENV PATH /usr/local/cuda/bin:$PATH
# ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Keep the container running
CMD ["sh", "-c", "while true; do sleep 3600; done"]