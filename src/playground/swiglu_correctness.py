import torch
import argparse
import gorby_swiglu
from torch.utils.benchmark import Timer
from typing import *


RED = 31
GREEN = 32
YELLOW = 33
BLUE = 34
MAGENTA = 35
CYAN = 36
WHITE = 37


def conditional_colorizer(condition: bool, text: str, colors: List) -> str:
	color = colors[0] if condition else colors[1]
	return f"\033[1;{color}m{text}\033[0m"


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
	"bfloat16": torch.bfloat16,
}


### Each entry under size is a Tuple with shape (AM,AN, BM, BN)
### A = [AM, AN]
### B = [BM, BN]
### C = [AM*BM, AN*BN]
workload_test_sizes = [
	(2, 256, 2, 256),
	(4, 256, 4, 256),
	(8, 256, 8, 256),
	(16, 256, 16, 256),
	(32, 256, 32, 256),
	(64, 256, 64, 256),
	(32, 768, 32, 768),
]


def torch_swiglu(A : torch.Tensor, B : torch.Tensor) -> torch.Tensor:
	return torch.nn.functional.silu(A) * B


if __name__ == "__main__":
	torch.manual_seed(37)

	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16", "float8"], default="float32")
	parser.add_argument("--check-runtime-only", action="store_true")
	args = parser.parse_args()

	device = torch.device(args.device)
	dtype = str_to_dtype_LUT[args.dtype]

	test_operator = gorby_swiglu.swiglu
	test_pass_comparison_list = []
	failed_test_pass_shapes = []


	for (AM, AN, BM, BN) in workload_test_sizes:
		### Tensor Init
		A = torch.randn(AM, AN, device=device, dtype=dtype)
		B = torch.randn(BM, BN, device=device, dtype=dtype)

		print(f"{'='*48}")
		print(f"A = [{A.shape[0]},{A.shape[1]}] | B = [{B.shape[0]},{B.shape[1]}]")
		print(f"{'='*48}")

		### Invoke test and baseline
		test_result = test_operator(A, B)
		torch_result = torch_swiglu(A, B)
		
		print(f"Test: {test_result}")
		print(f"Reference: {torch_result}")

		if args.check_runtime_only:
			pass
		else:
			numerically_correct = torch.allclose(test_result, torch_result)
			test_pass_comparison_list.append(numerically_correct)
			text = conditional_colorizer(numerically_correct, f"{numerically_correct}", [GREEN, RED])
			
			print(f"All Close? {text}")

			if not numerically_correct:
				failed_test_pass_shapes.append((AM, AN, BM, BN))

		print(f"\n{'='*48}")
	

	### Summarize Results
	if not args.check_runtime_only:
		all_passed = all(test_pass_comparison_list)
		text = conditional_colorizer(all_passed, f"{all_passed}", [GREEN, RED])

		print(f"Overall Pass? {text}")
		print(f"{'='*48}")

		for (AM, AN, BM, BN) in failed_test_pass_shapes:
			print(f"Failed Workload: A=[{AM},{AN}], B=[{BM},{BN}]")
