# Fine-Tuning Qwen2.5-7B on AMD Developer Cloud

Complete tutorial for fine-tuning Qwen2.5-7B using LoRA on AMD ROCm GPUs.

## 🚀 Quick Start on AMD Developer Cloud

### 1. Get AMD Developer Cloud Access
- Sign up at [developer.amd.com](https://developer.amd.com)
- Request MI250/MI300X instance access
- Launch ROCm-enabled instance

### 2. Setup Environment
```bash
# Upload this repo to your instance or clone it
git clone <your-repo-url>
cd fine-tune-llm

# Install Python (if 'python' command not found)
sudo apt update
sudo apt install python-is-python3

# Install conda (if not available)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment
conda env create -f environment.yml
conda activate llama3-rocm
```

### 3. Verify ROCm Setup
```bash
# Set ROCm environment variables
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# If PyTorch doesn't detect ROCm, reinstall with correct index
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

# Verify ROCm is working
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
rocm-smi  # Check GPU status
```

### 4. Authenticate with Hugging Face
```bash
# Install and login to Hugging Face
pip install huggingface_hub
huggingface-cli login
```

### 5. Prepare Your Data
Edit `data/sample.jsonl` with your instruction-output pairs:
```json
{"instruction": "Your question here", "output": "Expected response here"}
```

### 5. Run Fine-tuning
```bash
# For MI250 (start small)
python scripts/finetune.py \
  --model Qwen/Qwen2.5-7B \
  --data data/sample.jsonl \
  --output_dir results/checkpoints \
  --batch_size 1 \
  --epochs 1

# For MI300X (can handle larger batches)
python scripts/finetune.py \
  --model Qwen/Qwen2.5-7B \
  --data data/sample.jsonl \
  --output_dir results/checkpoints \
  --batch_size 4 \
  --epochs 1
```

### 6. Test Your Model
```bash
python scripts/infer.py \
  --model Qwen/Qwen2.5-7B \
  --lora results/checkpoints \
  --prompt "Your test prompt here"
```

### 7. Monitor Training
```bash
# Watch GPU usage
watch -n 1 rocm-smi

# Check system resources
htop
```

## 📊 Expected Performance

| GPU    | Batch Size | Training Time | Memory Usage |
|--------|------------|---------------|--------------|
| MI250  | 1-2        | 2-4 hours     | ~32GB        |
| MI300X | 4-8        | 1-2 hours     | ~64GB        |

## 💡 Tips for AMD Developer Cloud

- **Save your work**: Download results before instance expires
- **Monitor costs**: Instances charge by the hour
- **Start small**: Use batch_size=1 first, then increase
- **Check logs**: Training progress saved in `results/logs/`

---

## 📁 Repository Structure

```
fine-tune-llm/
├── README.md
├── environment.yml
├── docker/
│   └── Dockerfile
├── scripts/
│   ├── finetune.py      # Main training script
│   ├── infer.py         # Inference script
│   ├── benchmark.py     # Performance testing
│   └── prepare_data.py  # Data preprocessing
├── data/
│   └── sample.jsonl     # Example training data
├── notebooks/
│   └── Finetune_Llama3_ROCm.ipynb
└── results/
    ├── logs/
    ├── checkpoints/     # Saved models go here
    └── samples/
```

---

## 🔧 Scripts Overview

### finetune.py
Main training script with LoRA configuration:
- Uses 8-bit LoRA with rank=8
- Targets q_proj and v_proj layers
- Supports batch size adjustment for different GPUs

### infer.py
Test your fine-tuned model:
- Loads base model + LoRA weights
- Generates responses to prompts
- Configurable temperature and max tokens

### benchmark.py
Performance testing utility:
- Measures tokens/second
- Memory usage monitoring
- Multiple run averaging

---

## 🐳 Optional: Docker Setup

```bash
# Build container
docker build -t llama3-rocm -f docker/Dockerfile .

# Run with ROCm support
docker run --device=/dev/kfd --device=/dev/dri -it llama3-rocm
```

---

## 🛠 Troubleshooting

**ROCm not detected:**
```bash
# Check ROCm installation
/opt/rocm/bin/rocminfo
```

**Out of memory:**
- Reduce batch_size to 1
- Use gradient_checkpointing=True
- Try fp16 instead of bf16

**Slow training:**
- Increase batch_size if memory allows
- Use multiple GPUs with device_map="auto"
- Monitor GPU utilization with rocm-smi
