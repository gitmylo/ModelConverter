import argparse

import safetensors
import safetensors.torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    modes = parser.add_subparsers(dest="mode")

    file_mode = modes.add_parser("file", help="Inspect the whole tensor file")
    file_mode.add_argument("filemode", choices=("info", "meta")) # The operation to perform on the file
    # info = get tensor count
    # meta = get metadata

    tensor_mode = modes.add_parser("tensor", help="Inspect a tensor within the file")
    tensor_mode.add_argument("tensor")  # The full name of the tensor to inspect
    tensor_mode.add_argument("tensormode", choices=("printraw", "shape", "type", "asutf"))
    # printraw = print the raw tensor to stdout
    # asutf = read the tensor as an utf string and print it out

    args = parser.parse_args()
    inspect(args)

def inspect(args):
    with safetensors.safe_open(args.file, "pt") as tensors:
        if args.mode == "file":
            if args.filemode == "info":
                counter = 0
                counter_per_type = {}
                suffixes = {}
                for key in tensors.keys():
                    slice = tensors.get_slice(key)
                    dtype = slice.get_dtype()
                    suffix = key.split(".")[-1]
                    print(f"{key}: {str(slice.get_shape()).replace("[", "(").replace("]", ")")} [{dtype}]")
                    counter += 1
                    if dtype in counter_per_type:
                        counter_per_type[dtype] += 1
                    else:
                        counter_per_type[dtype] = 1
                    if suffix in suffixes:
                        suffixes[suffix] += 1
                    else:
                        suffixes[suffix] = 1
                print("==============================")
                print("Tensor count: " + str(counter))
                print("Per dtype: " + str(counter_per_type))
                print("Per suffix: " + str(suffixes))
            elif args.filemode == "meta":
                print(f"{tensors.metadata()}")
        elif args.mode == "tensor":
            tensor = tensors.get_tensor(args.tensor)
            if args.tensormode == "printraw":
                print(f"{tensor}")
            elif args.tensormode == "shape":
                print(f"{tensor.shape}")
            elif args.tensormode == "type":
                print(f"{tensor.dtype}")
            elif args.tensormode == "asutf":
                print(f"{bytes(tensor.tolist()).decode("utf-8")}")


if __name__ == "__main__":
    main()