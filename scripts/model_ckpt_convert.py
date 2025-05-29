import torch
from safetensors.torch import save_file

dir = "/root/align-anything/outputs/qwen_2_5_rm_trsize5k_5ep/slice_end/"
weights = torch.load(dir + "pytorch_model.bin")
save_file(weights, dir + "model.safetensors")  # 转为 SafeTensors