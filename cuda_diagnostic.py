import torch
import os
print("--------------------cuda_diagnostic.py-------------------")
print("pytorch version:",torch.__version__)
print("pytorch cuda version:",torch.version.cuda)

# Check if CUDA is available

if torch.cuda.is_available():
    print("CUDA is available on this system.")
else:
    print("CUDA is not available on this system.")

# Instantiate a CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("\n CUDA Version: \n")
try:    
    print(os.popen('nvcc --version').read())
except:
    print("CUDA is not present on the system path")
print("---------------------------------------------------------")

"""
(from copilot)
There could be several reasons why torch.cuda.is_available() is returning False:

PyTorch version: You might have installed the CPU version of PyTorch. Make sure you have installed the PyTorch version that supports CUDA.

CUDA version: Your CUDA version might not be compatible with the installed PyTorch version. Check the PyTorch website to see which CUDA versions are compatible with your PyTorch version.

GPU compatibility: Not all GPUs are CUDA-compatible. Make sure your NVIDIA GPU is CUDA-compatible.

Driver issues: Your NVIDIA drivers might not be up to date or might not be installed correctly. Try updating or reinstalling your drivers.

Operating System: Make sure your operating system supports CUDA. Some versions of Windows, for example, do not support CUDA.

Environment Variables: Sometimes, the PATH for CUDA might not be set properly in the environment variables.
"""