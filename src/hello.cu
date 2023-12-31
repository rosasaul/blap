
#include <stdio.h>
#include <stdlib.h>
#define N 512

__global__ void add(int *a, int *b, int *c) {
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// CPU function to generate a vector of random integers
void random_ints (int *a, int n) {
    for (int i = 0; i < n; i++)
        a[i] = rand() % 10000; // random number between 0 and 9999
}


int main(void) {
  int *a, *b, *c;              // host copies of a, b, c
  int *d_a, *d_b, *d_c;      // device copies of a, b, c
  int size = N * sizeof(int);
              
  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Alloc space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Alloc space for host copies of a, b, c and setup input values
  a = (int *)malloc(size); random_ints(a, N);
  b = (int *)malloc(size); random_ints(b, N);
  c = (int *)malloc(size);
  
  // Copy inputs to device
  cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU
  add<<<N,1>>>(d_a, d_b, d_c);

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  
  return 0;
}

