FROM pytorch/pytorch:latest
LABEL authors="soder"

RUN pip install transformers accelerate evaluate scikit-learn bitsandbytes Django

CMD ["sh", "-c", "while true; do sleep 3600; done"]