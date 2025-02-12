#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


int main() {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;

    // Initialize CUDA driver API
    cuInit(0);
    
    // Get handle for device 0
    cuDeviceGet(&cuDevice, 0);

    // Create context
    cuCtxCreate(&cuContext, 0, cuDevice);

    // Load the module from the cubin file
    if (cuModuleLoad(&cuModule, "ptx_inspect_source.cubin") != CUDA_SUCCESS) {
        std::cerr << "Failed to load module\n";
        return 1;
    }

    // Get the function handle from the module
    if (cuModuleGetFunction(&cuFunction, cuModule, "_ptx_simple_fmaf") != CUDA_SUCCESS) {
        std::cerr << "Failed to get function\n";
        return 1;
    }

    // Prepare data
    float *d_input, *d_output;
    constexpr int size = 1024; // Example size, adjust as needed
	float h_input[size] = {0};
	float h_output[size] = {0};


    // Initialize host input data
    for (int i = 0; i < size; ++i) h_input[i] = static_cast<float>(i);

    // Allocate memory on device
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Setup the kernel parameters
    void *args[] = { &d_input, &d_output };
    int blockSize = 256; // Example block size, adjust as needed
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch the kernel
    if (cuLaunchKernel(cuFunction, 
                       gridSize, 1, 1,  // grid dim
                       blockSize, 1, 1, // block dim
                       0, NULL, args, NULL) != CUDA_SUCCESS) {
        std::cerr << "Failed to launch kernel\n";
        return 1;
    }

    // Copy output back to host
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

	// Print output
	for (int i = 0; i < size; ++i) {
		std::cout << h_output[i] << " ";
	}

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cuModuleUnload(cuModule);
    cuCtxDestroy(cuContext);

    return 0;
}