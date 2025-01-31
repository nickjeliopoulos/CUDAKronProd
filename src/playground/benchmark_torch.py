import torch
torch.manual_seed(37)
import argparse
from torch.utils.benchmark import Timer


def measure_median_latency(func, *args, **kwargs):
	timer = Timer(
		stmt='func(*args, **kwargs)',
		globals={'func': func, 'args': args, 'kwargs': kwargs}
	)
	median_latency = timer.timeit(128).median * 1e3
	return median_latency


str_to_dtype_LUT = {
	"float32": torch.float32,
	"float16": torch.float16,
	"bfloat16": torch.bfloat16
}


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32")
	parser.add_argument("--M", type=int, default=4)
	parser.add_argument("--N", type=int, default=256)
	args = parser.parse_args()

	device = torch.device(args.device)
	dtype = str_to_dtype_LUT[args.dtype]

	A = torch.randn(args.M, args.N, device=device, dtype=dtype)
	B = torch.randn(args.M, args.N, device=device, dtype=dtype)

	print(f"A.shape: {A.shape}")
	print(f"B.shape: {B.shape}")

	C = torch.kron(A, B)
	
	print(f"Torch Result: {C}")