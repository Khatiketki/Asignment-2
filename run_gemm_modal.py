import subprocess
import modal

app = modal.App("assignment2-gemm")

# Base CUDA dev image + build tools
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential")
    # Copy your local CUDA file into the image at build time (Mount is removed)
    .add_local_file("gemm_naive.cu", remote_path="/root/gemm_naive.cu")
)

@app.function(
    image=image,
    gpu="any",          # Modal will attach an NVIDIA GPU
    timeout=60 * 20,    # 20 minutes
)
def run_gemm():
    print("Compiling on Modal GPU worker...")
    subprocess.run(
        ["nvcc", "-O2", "-std=c++17", "/root/gemm_naive.cu", "-o", "/root/gemm"],
        check=True,
    )

    print("Running...")
    # Optional: pass M N K
    subprocess.run(["/root/gemm", "512", "512", "512"], check=True)

@app.local_entrypoint()
def main():
    run_gemm.remote()
