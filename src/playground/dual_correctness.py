import torch
import argparse
import gorby_dual
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


### Workload sizes for testing
### (D_in, D_out)
workload_test_sizes = [
	# (8, 8),
	# (16, 16),
	# (32, 32),
	# (32, 64),
	# (32, 128),
	# (64, 32),
	# (64, 64),
	# (64, 128),
	# (128, 32),
	# (128, 64),
	# (128, 128),
	(256, 256),
	# (256, 512),
	# (256, 1024),
	(1024, 1024),
	# (2048, 1024),
	# (4096, 1024),
	# (2048, 2048),
	# (4096, 4096),
]


def torch_dual_fc(x, W, V, b, c) -> torch.Tensor:
	B, D_in = x.shape
	_, D_out = W.shape

	C = torch.empty(size=(B, 2*D_out), device=x.device, dtype=x.dtype)

	### Left
	C[:,0:D_out] = torch.matmul(x, W) + b
	### Right
	C[:,D_out:2*D_out] = torch.matmul(x, V) + c

	return C


if __name__ == "__main__":
	torch.manual_seed(37)

	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16", "float8"], default="float32")
	parser.add_argument("--check-runtime-only", action="store_true")
	parser.add_argument("--timing-bench", action="store_true")
	args = parser.parse_args()
	device = torch.device(args.device)
	dtype = str_to_dtype_LUT[args.dtype]

	test_operator = gorby_dual.dual_fc
	test_pass_comparison_list = []
	failed_test_pass_shapes = []

	### Batch size
	B = 2
	b_scalar = 0.0
	c_scalar = 0.0
	b = torch.tensor(data=[b_scalar], device=device, dtype=dtype)
	c = torch.tensor(data=[c_scalar], device=device, dtype=dtype)

	for (D_in, D_out) in workload_test_sizes:
		### Tensor Init
		x = torch.randn(B, D_in, device=device, dtype=dtype)
		W = torch.randn(D_in, D_out, device=device, dtype=dtype)
		V = torch.randn(D_in, D_out, device=device, dtype=dtype)

		print(f"{'='*48}")
		print(f"x = [{x.shape[0]},{x.shape[1]}] | W = [{W.shape[0]},{W.shape[1]}] | V = [{V.shape[0]},{V.shape[1]}]")

		### Invoke test and baseline
		test_result = test_operator(x, W, V, b, c)
		torch_result = torch_dual_fc(x, W, V, b, c)
		
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
				failed_test_pass_shapes.append((D_in, D_out))
				print(f"MaxAbs Diff = {torch.max(torch.abs(test_result - torch_result)):.2e}")
	

	### Summarize Results
	all_passed = all(test_pass_comparison_list)
	text = conditional_colorizer(all_passed, f"{all_passed}", [GREEN, RED])


	print(f"{'='*48}")
	print(f"Overall Pass? {text}")
	# print(f"Pass Rate = {100 * sum(test_pass_comparison_list)/len(test_pass_comparison_list):.1f}%")
	print(f"{'='*48}")


	if args.timing_bench:
		D_in, D_out = workload_test_sizes[-1]
		x = torch.randn(B, D_in, device=device, dtype=dtype)
		W = torch.randn(D_in, D_out, device=device, dtype=dtype)
		V = torch.randn(D_in, D_out, device=device, dtype=dtype)

		test_latency = measure_median_latency(test_operator, x, W, V, b, c)
		torch_latency = measure_median_latency(torch_dual_fc, x, W, V, b, c)

		if test_latency < torch_latency:
			text = conditional_colorizer(True, "FASTER", [GREEN, RED])
		else:
			text = conditional_colorizer(False, "SLOWER", [GREEN, RED])
		
		print(f"{'='*48}")
		print(f"TIMING TEST")
		print(f"Test Latency: {test_latency:.2f} ms")
		print(f"Reference Latency: {torch_latency:.2f} ms")
		print(f"Test is {text} than Reference")
		print(f"{'='*48}")
