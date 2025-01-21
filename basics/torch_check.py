
import torch

# 检查 CUDA 是否可用
print("CUDA Available:", torch.cuda.is_available())

# 检查当前的 GPU 设备
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current CUDA Device:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))