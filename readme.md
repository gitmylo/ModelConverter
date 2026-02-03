# Modelconverter
Scripts for safetensors model conversion and editing, such as nvfp4 quantization (see `to_format.py`).

## to_format.py
### usage:
`python to_format.py path/to/model.safetensors path/to/modeldef.sc types...`

types can be the name of a pytorch type, or nvfp4/fp8_scaled for quant types.  
Multiple types are supported if the `.sc` config supports it. More info [here](sc.md). Example `.sc` configs can be found in `model_defs`.

### types
Supports loading regular pytorch types, comfy format `nvfp4` and `fp8_scaled`.  
Supports converting to the following types (and aliases):
* `fp64` - `double`
* `fp32` - `fp`, `float`, `full`
* `fp16` - `half`
* `bf16` - `brain`
* `fp8_e4m3fn` - `fp8_e4m3`, `fp8_e4`, `fp8`
* `fp8_e5m2` - `fp8_e5`
* `nvfp4` - `fp4` (using comfy kitchen)
* `fp8_scaled` - `scaled_fp8` (using comfy kitchen)

### additional features
* SVD creation from precision losses `use --svd [rank]` to create a corrector lora. (Note: effectiveness depends on implementation.)

## Credits
* [comfy kitchen](https://github.com/Comfy-Org/comfy-kitchen) - Used for nvfp4 and fp8 scaled quantization and dequantization.