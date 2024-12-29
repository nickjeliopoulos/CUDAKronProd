import numpy

### Constants
FAIL = "FAIL"
OK = "OK"

### This script is used to check if some type of code results in any out of bounds errors or accesses
### Mostly used for scratch work without having to run kernels back to back to debug
class OOBCheckerProblemSizeAndKernelConfig:
	def __init__(self, MA, NA, MB, NB, THREADS, name=""):
		self.MA = MA
		self.NA = NA
		self.MB = MB
		self.NB = NB
		self.MC = MA * MB
		self.NC = NA * NB
		self.THREADS = THREADS
		self.name = name

	def __call__(self):
		return self.MA, self.NA, self.MB, self.NB, self.MC, self.NC, self.THREADS

	def __repr__(self):
		return f"MA: {self.MA}, NA: {self.NA}, MB: {self.MB}, NB: {self.NB}, MC: {self.MC}, NC: {self.NC}, THREADS: {self.THREADS}"


class OOBCheckerArray1D:
	def __init__(self, size : int):
		self.size = size
		self.oob_indices = []

	def __call__(self, idx : int) -> str:
		if idx > self.size and idx >= 0:
			self.oob_indices.append(idx)
			return FAIL
		return OK

	def __len__(self):
		return len(self.oob_indices)


class OOBCheckerArray2D:
	def __init__(self, size_x : int, size_y : int):
		self.size_x = size_x
		self.size_y = size_y
		self.oob_indices = []

	def __call__(self, x : int, y : int) -> str:
		fail_cond1 = x < 0 or y < 0
		fail_cond2 = x >= self.size_x or y >= self.size_y
		fail = fail_cond1 or fail_cond2
		if fail:
			self.oob_indices.append((x, y))
			return FAIL
		return OK

	def __len__(self):
		return len(self.oob_indices)


def checker_from_numpy(array: numpy.ndarray):
	if array.ndim == 1:
		return OOBCheckerArray1D(array.shape[0])
	elif array.ndim == 2:
		return OOBCheckerArray2D(array.shape[0], array.shape[1])
	else:
		raise ValueError("Unsupported array dimension")


### List of configurations to test
oob_checker_configs_sm80_kronecker_smem_fill = [
	OOBCheckerProblemSizeAndKernelConfig(32, 32, 32, 256, 256),
]


### Function(s) to test 
if __name__ == "__main__":
	### Dummy Test
	MA, NA, MB, NB, MC, NC, THREADS = oob_checker_configs_sm80_kronecker_smem_fill[0]()

	### Numpy Arrays (which may or may not have actual data)
	A = numpy.zeros((MA, NA))
	B = numpy.zeros((MB, NB))
	C = numpy.zeros((MC, NC))

	A_oob_chkr = checker_from_numpy(A)
	B_oob_chkr = checker_from_numpy(B)
	C_oob_chkr = checker_from_numpy(C)

	### Dummy OOB Accesses
	print( A_oob_chkr(32, 0) )
	print( B_oob_chkr(197, 64) )
	print( C_oob_chkr(32, 256) )
