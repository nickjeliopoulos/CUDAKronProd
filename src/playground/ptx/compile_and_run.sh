set -e

nvcc --cubin -arch=sm_80 fmaf_manual.ptx -o fmaf_manual.cubin
nvcc -arch=sm_80 main.cu -o main -lcuda -lcudart -L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\lib\\x64"

./main &> output.txt