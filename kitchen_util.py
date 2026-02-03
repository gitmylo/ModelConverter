import json

from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout, TensorCoreNVFP4Layout
import comfy_kitchen as ck
import torch

def quantize_nvfp4_weights(t: torch.Tensor, name: str, lora: dict[str, torch.Tensor], rank: int | None) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, str]], dict[str, torch.Tensor]]:
    if t.dtype not in [torch.float32, torch.float16, torch.bfloat16]: # Convert fp8, note: no fp8 scaled tensor loading yet, only regular fp8 models supported here.
        t = t.half()
    qt = QuantizedTensor.from_float(t, "TensorCoreNVFP4Layout")
    if rank is not None:
        lora = lora | decompose_lora_from_diff(t - qt.dequantize(), name, rank)
    q_data = {"format": "nvfp4"}
    return {
        f"{name}": qt._qdata,
        f"{name}_scale": qt.params.block_scale,
        f"{name}_scale_2": qt.params.scale,
        f"{name.removesuffix("weight")}comfy_quant": torch.tensor(list(json.dumps(q_data).encode('utf-8')), dtype=torch.uint8)
    }, {name.removesuffix(".weight"): q_data}, lora

def quantize_fp8_scaled_weights(t: torch.Tensor, name: str, lora: dict[str, torch.Tensor], rank: int | None) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, str]], dict[str, torch.Tensor]]:
    if t.dtype not in [torch.float32, torch.float16, torch.bfloat16]: # Convert fp8, note: no fp8 scaled tensor loading yet, only regular fp8 models supported here.
        t = t.half()
    qt = QuantizedTensor.from_float(t, "TensorCoreFP8Layout")
    if rank is not None:
        lora = lora | decompose_lora_from_diff(t - qt.dequantize(), name, rank)
    q_data = {"format": "float8_e4m3fn"}
    return {
        f"{name}": qt._qdata,
        f"{name}_scale": qt.params.scale,
        f"{name.removesuffix("weight")}comfy_quant": torch.tensor(list(json.dumps(q_data).encode('utf-8')), dtype=torch.uint8)
        # f"{name}_scale_2": qt.params.scale
    }, {name.removesuffix(".weight"): q_data}, lora

def decompose_lora_from_diff(diff: torch.Tensor, name: str, rank: int) -> dict[str, torch.Tensor]:
    if not name.endswith(".weight"):
        return {}
    if rank < 1:
        raise ValueError("Lora rank must be 1 or greater")
    base_name = name.removesuffix(".weight")

    if diff.dim() != 2:
        raise ValueError(f"Only 2D tensors are supported. Got {diff.dim()}D.")

    U, S, Vh = torch.linalg.svd(diff.cuda().float(), full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    sqrt_S_r = torch.diag(torch.sqrt(S_r))

    lora_B = U_r @ sqrt_S_r
    lora_A = sqrt_S_r @ Vh_r

    alpha = torch.tensor(float(rank))

    return {
        f"{base_name}.lora_A.weight": lora_A,
        f"{base_name}.lora_B.weight": lora_B,
        f"{base_name}.alpha": alpha
    }

# Class for loading weights with quantized weight support
# TODO: Include input scale
# TODO: Maybe don't dequantize after access (reduce memory usage)
class QuantizedWeightLoader:
    def __init__(self, weights: dict[str, torch.Tensor], comfy_quant: str | None):
        self.weights = weights

        self.fp8_prefixes = [] # .scaled_fp8 format
        for weight in self.weights.keys():
            if weight.endswith(".scaled_fp8"):
                self.fp8_prefixes.append(weight.removesuffix("scaled_fp8"))

        self.quant_layers = {}
        if comfy_quant is not None:
            self.quant_layers = json.loads(comfy_quant)["layers"] # {"blocks.0.cross_attn.k": {"format": "nvfp4"}}, formats: nvfp4/float8_e4m3fn

    def __getitem__(self, key: str):
        keycompare = key.removesuffix(".weight")
        if keycompare in self.quant_layers.keys():
            quant_info = self.quant_layers[keycompare]
            format = quant_info["format"]
            if format == "float8_e4m3fn":
                if key.endswith(".weight"):
                    scale = None
                    if (keycompare + ".scale_weight") in self.weights.keys():
                        scale = keycompare + ".scale_weight"
                    elif (keycompare + ".weight_scale") in self.weights.keys():
                        scale = key.removesuffix(".weight") + ".weight_scale"
                    if (keycompare + ".comfy_quant") in self.weights.keys():
                        self.weights.pop(keycompare + ".comfy_quant")
                    if scale is not None:
                        weight = self.weights[key]
                        scale_weight = self.weights.pop(scale)
                        weight = ck.dequantize_per_tensor_fp8(weight, scale_weight)
                        self.weights[key] = weight
                        return weight

            elif format == "nvfp4": # TODO: Test if correct
                if key.endswith(".weight"):
                    block_scale = keycompare + ".weight_scale"
                    tensor_scale = keycompare + ".weight_scale_2"
                    weight = self.weights[key]
                    block_scale_weight = self.weights.pop(block_scale)
                    tensor_scale_weight = self.weights.pop(tensor_scale)
                    weight = ck.dequantize_nvfp4(weight, tensor_scale_weight, block_scale_weight)
                    self.weights[key] = weight
                    return weight
            else:
                raise NotImplementedError(f"Unknown format {format}, dequant not implemented.")
        for p in self.fp8_prefixes: # Classic scaling
            if key.startswith(p) and key.endswith(".weight"):
                scale = None
                if (key.removesuffix(".weight") + ".scale_weight") in self.weights.keys():
                    scale = key.removesuffix(".weight") + ".scale_weight"
                elif (key.removesuffix(".weight") + ".weight_scale") in self.weights.keys():
                    scale = key.removesuffix(".weight") + ".scale_weight"
                if scale is not None:
                    weight = self.weights[key]
                    scale_weight = self.weights.pop(scale)
                    weight = ck.dequantize_per_tensor_fp8(weight, scale_weight)
                    self.weights[key] = weight
                    return weight
        return self.weights[key] # Default

    # Returns keys excluding scales
    def keys(self):
        return [key for key in self.weights.keys() if not key.endswith(".scale_input")
                and not key.endswith(".scale_weight") and not key.endswith(".weight_scale")
                and not key.endswith(".weight_scale_2") and not key.endswith("scaled_fp8") and not key.endswith(".comfy_quant")]

    def pop(self, key: str, d=None):
        out = self[key]
        self.weights.pop(key, d)
        return out

    def to_dict(self):
        out = {}
        for weight in list(self.keys()):
            out[weight] = self.pop(weight)

        return out
