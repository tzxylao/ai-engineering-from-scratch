import sys
import time


def check_gpu():
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Run: uv pip install torch torchvision torchaudio")
        return

    print("=== GPU Check ===\n")
    print(f"PyTorch version: {torch.__version__}")

    # 双分支：兼容CUDA(N卡) / MPS(M系列Mac)
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    device = None

    if has_cuda:
        device = torch.device("cuda")
        print(f"CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")
        vram_gb = props.total_memory / 1e9
    elif has_mps:
        device = torch.device("mps")
        print(f"CUDA available: False (Mac Apple Silicon uses MPS instead)")
        print(f"MPS Metal GPU available: True")
        print(f"Chip: Apple M5 (10-core GPU)")
        # MPS共享系统内存，读取本机总内存
        import psutil

        total_ram = psutil.virtual_memory().total / 1024**3
        print(f"Shared Unified Memory: {total_ram:.1f} GB")
        vram_gb = total_ram
    else:
        print(f"CUDA available: False")
        print("\nNo hardware GPU acceleration detected. That's fine for most lessons.")
        print("For GPU-heavy lessons, use Google Colab (free).")
        return

    print("\n=== CPU vs GPU Benchmark ===\n")
    size = 4000

    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # CPU 运算测速
    start = time.time()
    _ = a @ b
    cpu_time = time.time() - start
    print(f"CPU matrix multiply ({size}x{size}): {cpu_time:.3f}s")

    # GPU(MPS/CUDA) 运算测速
    a_gpu = a.to(device)
    b_gpu = b.to(device)
    # MPS不需要synchronize，CUDA需要，做兼容判断
    if has_cuda:
        torch.cuda.synchronize()

    start = time.time()
    _ = a_gpu @ b_gpu
    if has_cuda:
        torch.cuda.synchronize()
    # MPS同步等待计算完成
    torch.mps.synchronize() if has_mps else None

    gpu_time = time.time() - start
    print(f"GPU matrix multiply ({size}x{size}): {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")

    # 估算FP16最大模型参数
    params_fp16 = vram_gb * 1e9 / 2
    params_billions = params_fp16 / 1e9
    print(f"\nEstimated max model size (fp16): ~{params_billions:.0f}B parameters")


if __name__ == "__main__":
    # 如需读取内存，先安装psutil
    try:
        import psutil
    except ImportError:
        import subprocess

        subprocess.run(["uv", "pip", "install", "psutil"])
        import psutil
    check_gpu()
