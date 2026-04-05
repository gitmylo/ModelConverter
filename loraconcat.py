import torch
from safetensors.torch import load_file, save_file


def merge_loras_concat(path1, path2, output_path, scale1=1.0, scale2=1.0):
    # Load tensors
    lora1 = load_file(path1)
    lora2 = load_file(path2)

    merged_dict = {}

    # Get all unique keys (layer names)
    all_keys = set(lora1.keys()).union(set(lora2.keys()))

    for key in all_keys:
        # If the key isn't in both, we can't easily concat; skip or copy
        if key not in lora1 or key not in lora2:
            merged_dict[key] = lora1.get(key, lora2.get(key))
            continue

        # We only want to concat the weight matrices
        if ".lora_B.weight" in key:
            # B matrices: Concatenate horizontally (dim 1)
            # Apply scaling here
            w1 = lora1[key] * scale1
            w2 = lora2[key] * scale2
            merged_dict[key] = torch.cat([w1, w2], dim=1)

        elif ".lora_A.weight" in key:
            # A matrices: Concatenate vertically (dim 0)
            w1 = lora1[key]
            w2 = lora2[key]
            merged_dict[key] = torch.cat([w1, w2], dim=0)

        elif "alpha" in key:
            # Keep alpha intact (assuming they are the same)
            merged_dict[key] = lora1[key].mul(2.)

        else:
            # For any other parameters (like mid-layers), default to lora1
            merged_dict[key] = lora1[key]

    save_file(merged_dict, output_path)
    print(f"Successfully merged into {output_path}")


# Usage
merge_loras_concat(
    path1=r"",
    path2=r"",
    output_path="concatenated_lora.safetensors",
    scale1=0.5,
    scale2=0.7
)