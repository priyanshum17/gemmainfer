import os
import sys
import shutil
import subprocess
from setuptools import setup, find_packages

# Guard torch import so packaging tooling can inspect metadata without build isolation failures
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ext_modules = []
cmdclass = {}

if HAS_TORCH:
    # Check for CUDA availability
    has_cuda = torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1"

    sources = ["csrc/pybind.cpp"]
    extra_compile_args = {"cxx": ["-O3", "-std=c++20"]}

    if sys.platform == "darwin":
        os.environ["ARCHFLAGS"] = "-arch arm64"
        try:
            sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"]).decode().strip()
            extra_compile_args["cxx"] += [
                "-isysroot", sdk_path,
                "-stdlib=libc++",
                f"-I{sdk_path}/usr/include/c++/v1",
            ]
        except Exception:
            pass

    if has_cuda:
        sources += [
            "csrc/rmsnorm.cu",
            "csrc/softcapping.cu",
            "csrc/marlin.cu",
        ]
        extra_compile_args["nvcc"] = [
            "-O3",
            "-std=c++20",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            # Compute architectures for RTX 3090 (Ampere sm_86) and RTX 4090 (Ada sm_89)
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
            "-DUSE_CUDA",
        ]
        extra_compile_args["cxx"].append("-DUSE_CUDA")
        extension_cls = CUDAExtension
    else:
        extension_cls = CppExtension

    ext_modules = [
        extension_cls(
            name="gemmainfer_csrc",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]
    use_ninja = shutil.which("ninja") is not None
    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=use_ninja)}

setup(
    name="gemmainfer",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
