
"""
查看系统调用，关注 pt的核心库 libtorch.so

strace python3 demo_libtorch.py   >&  demo_libtorch.log
"""

import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = x + 1
print(y)
