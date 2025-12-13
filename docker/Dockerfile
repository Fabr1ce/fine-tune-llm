FROM rocm/pytorch:latest

RUN pip install transformers datasets peft accelerate sentencepiece

WORKDIR /workspace
COPY . .
