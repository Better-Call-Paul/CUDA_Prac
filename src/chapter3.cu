#include <iostream>
#include <vector>
#include <cstdlib>


// 3.1b
__global__ void mat_add(int N, float *a, float *b, float *c) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        int index = row * N + col;
        c[index] = a[index] + b[index];
    }
}

// 3.1b
__global__ void mat_add_row(int N, float *a, float *b, float *c) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        for (int col = 0; col < N; col++) {
            int index = row * N + col;
            c[index] = a[index] + b[index];
        }
    }
}

// 3.1c
__global__ void mat_add_col(int N, float *a, float *b, float *c) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < N) {
        for (int row = 0; row < N; row++) {
            int index = col * N + row;
            c[index] = a[index] + b[index];
        }
    }
}

// 3.2
__global__ void vec_to_mat_multiply(int N, float *b, float *c, float *a) {
    int row = blockDim.x + blockIdx.x + threadIdx.x;

    if (row < N) {
        float sum = 0.0;
        for (int col = 0; col < N; col++) {
            sum += b[row * N + col] + c[col];
        }
        a[row] = sum;
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
                    (N + blockDimensions.x - 1) / blockDimensions.x,
                    (N + blockDimensions.y - 1) / blockDimensions.y
    );

    mat_add<<<blockQuantity, blockDimensions>>>(N, a, b, c);
    
    cudaDeviceSynchronize();
    cudaMemcpy(host_c.data(), c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    std::cout << "Mat Add Complete " << "\n";


    // 3.1

    // 3.2
    /*
    size_t block = size * sizeof(float);

    float *host_a = (float *)malloc(sizeof(float) * block);
    vector<float> host_b(size);
    float *host_c = (float *)malloc(sizeof(float) * block);

    for (int i = 0; i < N * N) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *a, *b, *c;

    cudaMalloc(&a, block);
    cudaMalloc(&b, block);
    cudaMalloc(&c, block);

    cudaMemcpy(a, host_a, block, cudaMemcpyHostToDevice);
    cudaMemcpy(b, host_b.data(), block, cudaMemcpyHostToDevice);


    vec_to_mat_multiply<<<blockQuantity, blockDimensions>>>(N, host_b, host_c, host_a);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    std::cout << "Mat Mul Done " << "\n";
    */
    //


    return 0;
}