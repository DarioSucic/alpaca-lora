## 🦙🌲🤏 Alpaca-LoRA: Low-Rank LLaMA Instruct-Tuning

This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) results using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).

In addition to the training code, which runs within five hours on a single RTX 4090,
we publish a script for downloading and inference on the foundation model and LoRA.
To fine-tune cheaply and efficiently, we use Huggingface's [PEFT](https://github.com/huggingface/peft)
as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

Without hyperparameter tuning or validation-based checkpointing, the LoRA model produces outputs comparable to the Stanford Alpaca model. (Please see the outputs included below.) Further tuning might be able to achieve better performance; I invite interested users to give it a try and report their results.

### Setup
```
pip install -q datasets loralib sentencepiece

pip uninstall transformers
pip install -q git+https://github.com/zphang/transformers@c3dc391

pip install -q git+https://github.com/huggingface/peft.git
```

#### `bitsandbytes` warning
Adam 8bit training is broken on `bitsandbytes` versions after 35.0 and until at least 37.0, and will cause loss to explode to inf.
The workaround is to either use 35.0, a fixed future version, or using another optimizer (such as Lion below).

#### (Optional) Lion optimizer
To use the Lion optimizer we can use a `bitsandbytes` fork by LucidRains.

```
git clone git@github.com:lucidrains/bitsandbytes.git
cd bitsandbytes
export CUDA_VERSION=118
make cuda11x && pip install .
```

We can optionally build `bitsandbytes` faster with a smaller binary by only compiling for our desired gpu arch:

```
+CC_ADA := -gencode arch=compute_89,code=sm_89
+cuda11x_ada: $(BUILD_DIR) env
+	$(NVCC) $(CC_ADA) -Xcompiler '-fPIC' --use_fast_math -Xptxas=-v -dc $(FILES_CUDA) $(INCLUDE) $(LIB) --output-directory $(BUILD_DIR)
+	$(NVCC) $(CC_ADA) -Xcompiler '-fPIC' -dlink $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o -o $(BUILD_DIR)/link.o
+	$(GPP) -std=c++14 -DBUILD_CUDA -shared -fPIC $(INCLUDE) $(BUILD_DIR)/ops.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/link.o $(FILES_CPP) -o ./bitsandbytes/libbitsandbytes_cuda$(CUDA_VERSION).so $(LIB)
```

### Inference (`generate.py`)
This file reads the foundation model from the Huggingface model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

### Training (`train.py`)
For single GPU training, check out `train_single_gpu.sh`. For multi-GPU using PyTorch FSDP, look at `train_fsdp.sh`.


### Errors
> CUDA SETUP: WARNING! libcuda.so not found! Do you have a CUDA driver installed? If you are on a cluster, make sure you are on a CUDA machine!

For WSL2 users, this can be fixed by symlinking `libcuda.so` properly ([src](https://forums.developer.nvidia.com/t/wsl2-libcuda-so-and-libcuda-so-1-should-be-symlink/236301)):

```bash
cd /usr/lib/wsl/lib
sudo rm libcuda.so libcuda.so.1
sudo ln -s libcuda.so.1.1 libcuda.so.1
sudo ln -s libcuda.so.1 libcuda.so
sudo ldconfig
```
