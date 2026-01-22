import torch
# 1. 查看PyTorch版本和CUDA关联状态
print("PyTorch版本:", torch.__version__)
# 2. 检查CUDA是否可用（必须输出True！）
print("CUDA可用:", torch.cuda.is_available())
# 3. 查看CUDA版本
print("CUDA版本:", torch.version.cuda)
# 4. 查看GPU设备
print("GPU数量:", torch.cuda.device_count())
print("GPU型号:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")