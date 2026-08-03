import safetensors.torch

target = ""
apply = safetensors.torch.load_file(r"")
original = safetensors.torch.load_file(r"")

if target not in apply or target not in original:
    raise ValueError("Target not exist")

apply[target] = original[target]

safetensors.torch.save_file(apply, r"")
