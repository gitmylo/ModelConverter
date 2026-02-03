import sys

import safetensors.torch

file = sys.argv[1]
model = safetensors.torch.load_file(file)
qmeta = safetensors.safe_open(file, "pt").metadata().get("_quantization_metadata", None)

print(", ".join(model.keys()))
print(qmeta)