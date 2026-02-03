import argparse
import json

import safetensors.torch
import torch
from tqdm import tqdm

from kitchen_util import quantize_nvfp4_weights, quantize_fp8_scaled_weights, decompose_lora_from_diff, \
    QuantizedWeightLoader
from sc_parser import build_sc_tree


def main():
    args = parse_args()

    target_format = [name_to_torch_format(f) for f in args.format]
    input_file = args.input

    compress_rules = args.rules
    f = open(compress_rules)
    sc_rules = f.read()
    f.close()
    sc_tree = build_sc_tree(sc_rules)

    svd_rank = args.svd
    svd_lora_weight_prefix = args.svd_prefix

    def should_compress(param: str) -> bool | int:
        # return compress_funcs[compress_rules](param)
        result = sc_tree.match_str(param)
        if result is None:
            result = False
        return result

    format_name = str(target_format[0]).removeprefix("torch.").replace("float", "fp") + ("_mixed" if len(target_format) > 1 else "")
    output_file = ".".join(input_file.split(".")[:-1]) + f"_{format_name}.safetensors"
    output_lora = ".".join(input_file.split(".")[:-1]) + f"_{format_name}_svd-corrector_rank{svd_rank}.safetensors"
    in_meta = safetensors.safe_open(input_file, "pt").metadata()
    if in_meta is not None:
        in_meta = in_meta.get("_quantization_metadata", None)
    in_model = QuantizedWeightLoader(safetensors.torch.load_file(input_file), in_meta)
    out_model = {}
    quant_metadata = {}
    lora_model = {}

    targets = {key: response_to_type(should_compress(key), target_format) for key in in_model.keys()}
    targets = {key: targets[key] for key in targets if targets[key] is not None}

    prog = tqdm(targets.keys())
    prog.set_description_str(f"Converting to {format_name}")
    for key in prog:
        target_type = targets[key]
        target_name = str(target_type).removeprefix("torch.").replace("float", "fp")
        prog.set_description_str(f"Converting to {target_name}")
        prog.set_postfix_str(key)
        # whole_model[key] = whole_model[key].to(target_format)
        keys, meta, lora_model = to_format(in_model[key], key, target_type, lora_model, svd_rank)
        in_model.pop(key, None)  # Remove key
        out_model = out_model | keys  # Merge the keys
        quant_metadata = quant_metadata | meta  # Merge the metadata

    # for key in list(whole_model.keys()):
    #     if should_compress(key):
    #         # whole_model[key] = whole_model[key].to(target_format)
    #         keys, meta = to_format(whole_model[key], key, target_format)
    #         whole_model.pop(key, None) # Remove key in case of renaming
    #         whole_model = whole_model | keys # Merge the keys
    #         quant_metadata = quant_metadata | meta # Merge the metadata

    if len(quant_metadata) == 0:
        safetensors.torch.save_file(in_model.to_dict() | out_model, output_file)
    else:
        safetensors.torch.save_file(in_model.to_dict() | out_model, output_file,
                                    metadata={"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": quant_metadata})})
    if svd_rank is not None:
        if svd_lora_weight_prefix is not None:
            add_lora_prefix(lora_model, svd_lora_weight_prefix)
        safetensors.torch.save_file(lora_model, output_lora)

def response_to_type(response: int | bool, target_format) -> torch.dtype | str | None:
    if not isinstance(response, bool): # isinstance int will give false positives on bools
        return target_format[min(response, len(target_format)-1)]
    if response:
        return target_format[0]
    return None

def to_format(t: torch.Tensor, name: str, format: torch.dtype | str, lora: dict[str, torch.Tensor], svd_rank: int | None) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, str]], dict[str, torch.Tensor]]:
    if isinstance(format, torch.dtype):
        return to_torch_format(t, name, format, lora, svd_rank)
    match format.lower().replace("float", "fp"):
        case "nvfp4" | "fp4":
            return quantize_nvfp4_weights(t, name, lora, svd_rank)
        case "fp8_scaled" | "scaled_fp8":
            return quantize_fp8_scaled_weights(t, name, lora, svd_rank)

        # Torch type aliases
        case _:
            return to_torch_format(t, name, name_to_torch_format(format), lora, svd_rank)

def name_to_torch_format(name: str):
    match name.lower().replace("float", "fp"):
        case "fp8_e4m3fn" | "fp8_e4m3" | "fp8_e4" | "fp8":
            return torch.float8_e4m3fn
        case "fp8_e5m2" | "fp8_e5":
            return torch.float8_e5m2
        case "fp16" | "half":
            return torch.float16
        case "bf16" | "brain":
            return torch.bfloat16
        case "fp32" | "fp" | "full":
            return torch.float32
        case "fp64" | "double":
            return torch.float64
        # Non torch
        case "fp4":
            return "nvfp4"
        case "scaled_fp8":
            return "fp8_scaled"
    return name

def to_torch_format(t: torch.Tensor, name: str, format: torch.dtype, lora: dict[str, torch.Tensor], svd_rank: int | None) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, str]], dict[str, torch.Tensor]]:
    converted = t.to(format)
    if svd_rank is not None:
        converted_back = converted.to(t.dtype)
        lora = lora | decompose_lora_from_diff(t - converted_back, name, svd_rank)
    return {name: converted}, {}, lora

def add_lora_prefix(lora: dict[str, torch.Tensor], prefix: str):
    for key in list(lora.keys()):
        lora[prefix + key] = lora.pop(key)

def chroma_rad_sc(param: str) -> int | bool:
    if param.endswith(".weight") and not "distilled_guidance_layer" in param and not "nerf" in param:
        if "qkv" in param or "proj" in param:
            return 0
        if "mlp" in param:
            if "mlp.0." in param:
                return 0
            if "mlp.2." in param:
                return 1
        if "single_blocks" in param and "linear" in param:
            if "linear1" in param:
                return 0
            if "linear2" in param:
                return 1
    return False

# Low quality
def chroma_sc(param: str) -> int | bool:
    if param.endswith(".weight") and not "distilled_guidance_layer" in param:
        if "qkv" in param or "proj" in param:
            return 0
        if "mlp" in param:
            if "mlp.0." in param:
                return 0
            if "mlp.2." in param:
                return 1
        if "single_blocks" in param and "linear" in param:
            if "linear1" in param:
                return 0
            if "linear2" in param:
                return 1
    return False

# Good quality
def gemma3_sc(param: str) -> bool:
    if param.endswith(".weight"): # Skip biases
        if "mlp.fc" in param:
            return True
        if ("self_attn" in param or "mlp" in param) and "proj" in param:
            return True
    return False

# Good quality
def zimage_sc(param: str) -> bool:
    if param.endswith(".weight"):
        if "layers" in param:
            if "qkv" in param or "attention.out" in param:
                return True
            if "feed_forward.w" in param:
                return True
    return False

# Untested
def kandinsky_sc(param: str) -> int | bool:
    if param.endswith(".weight"):
        if "text_transformer_blocks" in param or "visual_transformer_blocks" in param:
            if "modulation.out_layer" in param: # text/visual modulation out layer
                return 0
            if "attention.to_" in param or "attention.out_layer" in param: # self/cross attention to/out layer
                if "attention.to_value" in param:
                    return 1
                return 0
            if "feed_forward" in param:
                if "out_layer" in param:
                    return 1
                return 0
    return False

# Decent quality
def wan_sc(param: str) -> int | bool:
    if param.endswith(".weight"):
        if "blocks." in param:
            if ".ffn.0." in param:
                return 0
            if ".ffn.2." in param:
                return 1
            if "_attn." in param:
                if ".q." in param or ".k." in param or ".o." in param:
                    return 0
                if ".v." in param:
                    return 1
    return False

def all_sc(param: str) -> bool:
    return True

compress_funcs = {
    "any": all_sc,
    "chroma rad": chroma_rad_sc,
    "chroma": chroma_sc,
    "gemma3": gemma3_sc,
    "z-image": zimage_sc,
    "kandinsky": kandinsky_sc,
    "wan": wan_sc
}

# input_file = ""
# target_format = torch.float8_e4m3fn
# target_format = ["nvfp4", torch.float8_e4m3fn]
# target_format = ["nvfp4"]
# target_format = [torch.float16]
# compress_rules = "wan"
# svd_rank: int | None = None # Use SVD to create a corrector lora, set to None to disable
# svd_lora_weight_prefix = "diffusion_model."

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="The model to convert.")
    parser.add_argument("rules", type=str, help="A file containing the rules for quantization.")
    parser.add_argument("format", type=str, nargs="+", help="The quantization format, for example fp4, fp8, fp8_e5m2, fp8_scaled, bf16. Multiple values supported for mixed quant.")
    parser.add_argument("--svd", type=int, default=None, help="Svd rank to use.")
    parser.add_argument("--svd-prefix", type=str, default="diffusion_model.", help="Custom prefix for svd lora weights.")

    return parser.parse_args()

if __name__ == "__main__":
    main()