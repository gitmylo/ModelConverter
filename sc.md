# .sc file format
The .sc (chould compress) file format indicates which weights should be compressed and which weights shouldn't.

The parser is very basic and might behave unexpectedly if rules aren't followed.

## Syntax
The file is read from top to bottom, if a rule passes, the next indented line is read, if a rule ends with `= v` where `v` is a value, the rule instead returns when passed.  
The indentation and logical operators allow for writing decision trees based on the weight name, while keeping it readable.


## Match rules
* `"name"` - Weight name must contains `name` to pass.
* `s"name"` - Weight name must start with `name` to pass.
* `e"name"` - Weight name must end with `name` to pass.

## Logical operators
* `"name" = 0` - Return `0` if the weight contains `name`.
* `0` - Returns `0`.
* `"name1" | "name2"` - Weight name must contain `name1` or `name2` to pass.
* `"name1" "name2"` - Weight name must contain `name1` and `name2` to pass.

Note that a line can not mix `or` and `and`, if `|` is detected on a line all strings on that line are viewed as different values for or.  
Same goes for `=`, it simply indicates that the last non-whitespace text should be parsed as a result.

## Return values
Valid values: x or any integer.  
`x` means false, don't compress this weight.  
`0` means compress this weight to the first indicated type.  
`1` means comrpess this weight to the second indicated type, or first if only one is given.  
Example, running with `fp4 fp8_scaled bf16` `0` will be quantized to `fp4`, `1` will be quantized to `fp8_scaled`, etc.

If the file ends without a return, `x` is assumed as the default, adding `x` to the end of the file is still recommended for clarity.


## Example
Chroma:
```
e".weight"
 "distilled_guidance_layer" = x
 "qkv" | "proj" = 0
 "mlp"
  "mlp.0." = 0
  "mlp.2." = 1
 "single_blocks" "linear"
  "linear1" = 0
  "linear2" = 1
x
```
This is comparable to the python code:
```python
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
```