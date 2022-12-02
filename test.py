import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
from torch.backends import cudnn
print(cudnn.is_available() )