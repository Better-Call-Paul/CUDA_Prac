#include <iostream>
#include <vector>
#include <cstdlib>



__global__ void mat_add(int N, float *a, float *b, float *c) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        int index = row * N + col;
        c[index] = a[index] + b[index];
    }
}

int main() {

    // 3.1

    int N = 16;
    size_t size = N * N;
    std::vector<float> host_a(size), host_b(size), host_c(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int index = i * N + j;
            host_a[index] = static_cast<float>(rand()) / RAND_MAX;
            host_b[index] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float *a, *b, *c;
    cudaMalloc(&a, size * sizeof(float));
    cudaMalloc(&b, size * sizeof(float));
    cudaMalloc(&c, size * sizeof(float));

    cudaMemcpy(a, host_a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, host_b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDimensions(16, 16);
    dim3 blockQuantity(
                    (blockDimensions.x + N - 1) / N,
                    (blockDimensions.y + N - 1) / N
    );

    mat_add<<<blockQuantity, blockDimensions>>>(N, a, b, c);
    
    cudaDeviceSynchronize();
    cudaMemcpy(host_c.data(), &c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    std::cout << "Mat Add Complete " << "\n";


    // 3.1



    


    return 0;
}