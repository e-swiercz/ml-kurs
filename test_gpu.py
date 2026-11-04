import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))  # oczekiwane (12, 0)
a = torch.randn(4096, 4096, device="cuda")
print((a @ a).sum().item())
