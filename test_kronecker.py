import torch
torch.manual_seed(37)
import argparse
from gorby_kronecker import kronecker_product as gorby_kronecker_product


str_to_dtype_LUT = {
	"float32": torch.float32,
	"float16": torch.float16,
	"bfloat16": torch.bfloat16
}


def torch_kronecker_product(A : torch.Tensor, B : torch.Tensor) -> torch.Tensor:
	return torch.kron(A, B)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32")
	parser.add_argument("--M", type=int, default=None)
	parser.add_argument("--N", type=int, default=None)
	args = parser.parse_args()

	device = torch.device(args.device)
	dtype = str_to_dtype_LUT[args.dtype]

	if args.M is not None and args.N is not None:
		A = torch.randn(args.M, args.M, device=device, dtype=dtype)
		B = torch.randn(args.N, args.N, device=device, dtype=dtype)
	else:
		A = torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device)
		B = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=dtype, device=device)

	reference = torch_kronecker_product(A, B)
	test = gorby_kronecker_product(A, B)

	print(f"Reference:\n{reference}")
	print(f"Test:\n{test}")
	print(f"All Close?: {torch.allclose(reference, test)}")
