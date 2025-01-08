import torch
torch.manual_seed(37)
import argparse
from gorby_kronecker import kronecker_tiny_product, kronecker_anyrow_product, kronecker_anyrow_smem_product, kronecker_anyrow_anycol_product
from torch.utils.benchmark import Timer
from typing import List, Dict, Any, Tuple


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
	"bfloat16": torch.bfloat16
}

str_to_kron_variant_callable_LUT = {
	"tiny": kronecker_tiny_product,
	"anyrow": kronecker_anyrow_product,
	"anyrow_smem": kronecker_anyrow_smem_product,
	"anyrow_anycol": kronecker_anyrow_anycol_product,
	"adaptive": None
}

### Each entry under size is a Tuple with shape (AM,AN, BM, BN)
workload_test_sizes = [
	###
	(2, 512, 4, 512),
	(3, 512, 4, 512),
	(4, 512, 4, 768),
	(7, 512, 4, 768),
	(16, 512, 16, 512),
	(8, 512, 8, 512),
	(4, 512, 4, 512),
	### 
	# (16, 512, 24, 768),
	# (4, 512, 4, 1024),
	# (18, 512, 4, 1024),
	# (23, 512, 4, 1024),
	# (32, 512, 8, 1024),
]


def torch_kronecker_product(A : torch.Tensor, B : torch.Tensor) -> torch.Tensor:
	return torch.kron(A, B)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32")
	parser.add_argument("--check-runtime-only", action="store_true")
	parser.add_argument("--kron-variant", type=str, choices=["tiny", "anyrow", "anyrow_smem", "anyrow_anycol", "adpative"], default="anyrow_anycol")
	args = parser.parse_args()

	device = torch.device(args.device)
	dtype = str_to_dtype_LUT[args.dtype]

	test_operator = str_to_kron_variant_callable_LUT[args.kron_variant]
	test_pass_comparison_list = []
	failed_test_pass_shapes = []

	for (AM, AN, BM, BN) in workload_test_sizes:
		A = torch.randn(AM, AN, device=device, dtype=dtype)
		B = torch.randn(BM, BN, device=device, dtype=dtype)

		# print(f"A.shape: {A.shape}")
		# print(f"B.shape: {B.shape}")

		print(f"{'='*48}")
		print(f"A = [{A.shape[0]},{A.shape[1]}] | B = [{B.shape[0]},{B.shape[1]}] | C = [{A.shape[0]*B.shape[0]},{A.shape[1]*B.shape[1]}]")
		print(f"{'='*48}")

		test_result = test_operator(A, B)
		torch_result = torch_kronecker_product(A, B)
		
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

		print(f"{'='*48}")

	if not args.check_runtime_only:
		all_passed = all(test_pass_comparison_list)
		text = conditional_colorizer(all_passed, f"{all_passed}", [GREEN, RED])
		print(f"Overall Pass? {text}")
		print(f"{'='*48}")
		for (AM, AN, BM, BN) in failed_test_pass_shapes:
			print(f"Failed Workload: A=[{AM},{AN}], B=[{BM},{BN}]")
