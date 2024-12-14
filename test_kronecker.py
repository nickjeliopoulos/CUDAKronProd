import torch
from gorby_kronecker import kronecker_product as gorby_kronecker_product


def torch_kronecker_product(A, B):
	return torch.kron(A, B)


if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	A = torch.tensor([[1, 1], [1, -1]], dtype=torch.float32, device=device)
	B = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32, device=device)
	print(torch_kronecker_product(A, B))

	print(gorby_kronecker_product(A, B))
