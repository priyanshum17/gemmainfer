#!/usr/bin/env bash
# ==============================================================================
# GemmaInfer Turnkey GPU Deployment Script (RTX 3090 / 4090 / Any Cloud GPU)
# ==============================================================================
# Sets up Python virtual environment, installs CUDA PyTorch with uv, compiles native
# C++/CUDA kernels with Ninja, verifies hardware correctness, and launches the engine.
#
# Usage:
#   bash scripts/deploy_gpu.sh          # Interactive deployment menu
#   bash scripts/deploy_gpu.sh verify   # Run 5-stage verification only
#   bash scripts/deploy_gpu.sh chat     # Launch interactive CLI chat
#   bash scripts/deploy_gpu.sh serve    # Launch OpenAI-compatible HTTP server
#   bash scripts/deploy_gpu.sh download # Download companion assistant model
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=================================================================="
echo " GemmaInfer Turnkey GPU Deployment (RTX 3090 / 4090 / 24GB VRAM) "
echo "=================================================================="

# 1. Check NVIDIA Driver & GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. Ensure NVIDIA GPU and drivers are installed."
    exit 1
fi

echo -n "Checking NVIDIA Hardware: "
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
echo "${GPU_NAME} (${GPU_VRAM} MiB VRAM, Driver ${DRIVER_VER})"

# 2. Install uv package manager if missing
if ! command -v uv &> /dev/null; then
    echo "[INFO] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

# 3. Setup Virtual Environment (Python 3.12 / 3.11 with graceful fallback)
VENV_DIR="${REPO_ROOT}/.venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "[INFO] Creating virtual environment at ${VENV_DIR}..."
    uv venv "${VENV_DIR}" --python 3.12 2>/dev/null || uv venv "${VENV_DIR}" || python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

# 4. Install Dependencies (Ninja + PyTorch + Requirements)
echo "[INFO] Installing PyTorch with CUDA support and build dependencies..."
export UV_HTTP_TIMEOUT=300
uv pip install ninja
uv pip install torch
uv pip install --no-build-isolation -e ".[dev,serve]"

# 5. Compile C++/CUDA Kernels
echo "[INFO] Compiling native CUDA kernels with Ninja (Ampere sm_86 & Ada sm_89)..."
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.6;8.9"
python setup.py build_ext --inplace

# Verify compiled library exists
if ! ls gemmainfer_csrc*.so 1> /dev/null 2>&1; then
    echo "[ERROR] CUDA kernel compilation failed. Check compiler logs above."
    exit 1
fi
echo "[PASS] Native CUDA kernels compiled successfully (gemmainfer_csrc*.so)."

# Model paths configuration (environment variable or default ./models directory)
MODEL_DIR="${GEMMA_MODEL_DIR:-${REPO_ROOT}/models}"
BASE_MODEL="${GEMMA_MODEL_PATH:-${MODEL_DIR}/gemma-4-26B-A4B-it-W4A16}"
if [ ! -d "${BASE_MODEL}" ] && [ -d "/root/models/gemma-4-26B-A4B-it-W4A16" ]; then
    BASE_MODEL="/root/models/gemma-4-26B-A4B-it-W4A16"
fi

ASSISTANT_MODEL="${GEMMA_ASSISTANT_PATH:-${MODEL_DIR}/gemma-4-26B-A4B-it-assistant}"
if [ ! -d "${ASSISTANT_MODEL}" ] && [ -d "/root/models/gemma-4-26B-A4B-it-assistant" ]; then
    ASSISTANT_MODEL="/root/models/gemma-4-26B-A4B-it-assistant"
fi

run_chat() {
    if [ -d "${BASE_MODEL}" ]; then
        echo "[INFO] Launching interactive CLI with weights: ${BASE_MODEL}"
        python scripts/chat.py --model-path "${BASE_MODEL}" --device cuda
    else
        echo "[INFO] No weights found at ${BASE_MODEL}. Launching with synthetic Gemma 4 26B A4B..."
        python scripts/chat.py --synthetic --device cuda
    fi
}

run_serve() {
    ARGS="--host 0.0.0.0 --port 8000 --device cuda"
    if [ -d "${BASE_MODEL}" ]; then
        echo "[INFO] Launching server with weights: ${BASE_MODEL}"
        ARGS="${ARGS} --model-path ${BASE_MODEL}"
        if [ -d "${ASSISTANT_MODEL}" ]; then
            echo "[INFO] Attaching companion assistant: ${ASSISTANT_MODEL}"
            ARGS="${ARGS} --assistant-model-path ${ASSISTANT_MODEL}"
        fi
    else
        echo "[INFO] No weights found at ${BASE_MODEL}. Launching with synthetic Gemma 4 26B A4B..."
        ARGS="${ARGS} --synthetic"
    fi
    python scripts/serve.py ${ARGS}
}

download_models() {
    mkdir -p "${MODEL_DIR}"
    echo "[INFO] Installing huggingface_hub..."
    uv pip install -q "huggingface_hub[cli]"
    echo "[INFO] Downloading GotoAI-Inc/gemma-4-26B-A4B-it-W4A16 base checkpoint..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('GotoAI-Inc/gemma-4-26B-A4B-it-W4A16', local_dir='${BASE_MODEL}')"
    echo "[INFO] Downloading official Google Gemma companion assistant model..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-26B-A4B-it-assistant', local_dir='${ASSISTANT_MODEL}')"
    echo "[INFO] Both models ready at ${BASE_MODEL} and ${ASSISTANT_MODEL}."
}

# 6. Action Selection
ACTION="${1:-prompt}"

if [ "${ACTION}" = "verify" ]; then
    echo "[INFO] Running 5-stage GPU hardware & numerical verification..."
    python scripts/verify_gpu.py
    echo "[INFO] Verification complete. Exiting."
    exit 0
elif [ "${ACTION}" = "chat" ]; then
    run_chat
elif [ "${ACTION}" = "serve" ]; then
    run_serve
elif [ "${ACTION}" = "download" ]; then
    download_models
elif [ "${ACTION}" = "prompt" ]; then
    echo ""
    echo "=================================================================="
    echo " GemmaInfer is fully deployed on your GPU!"
    echo " Select an option to start:"
    echo "   1) Interactive CLI Chat (Terminal generation with token streaming)"
    echo "   2) OpenAI-Compatible HTTP Server (Port 8000 for OpenWebUI / curl)"
    echo "   3) Latency & Throughput Benchmark Suite"
    echo "   4) Speculative Decoding Verification Benchmark"
    echo "   5) Download Official Companion Assistant Model"
    echo "   6) Exit"
    echo "=================================================================="
    read -r -p "Enter choice [1-6]: " CHOICE
    case "${CHOICE}" in
        1)
            run_chat
            ;;
        2)
            run_serve
            ;;
        3)
            python scripts/benchmark.py --synthetic --device cuda
            ;;
        4)
            python scripts/test_speculative.py --synthetic --device cuda
            ;;
        5)
            download_models
            ;;
        *)
            echo "Exiting. You can launch anytime with: source .venv/bin/activate && python scripts/serve.py"
            ;;
    esac
fi
