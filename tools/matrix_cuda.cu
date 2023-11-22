
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <chrono>
#define N 512
#define BLOCK_SIZE 16

using namespace std;

__global__ void gpu_vecMatrix_mult(double *matrix, double *input, double *output, int matrix_size){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x;

  //printf("blockIdx.x %d blockDim.x %d threadIdx.x %d\n",blockIdx.x,blockDim.x,threadIdx.x);

  double sum = 0;
  for(int i = 0; i < matrix_size; i++){
    sum += matrix[index * matrix_size + i] * input[i];
  }
  output[index] = sum;
}

__global__ void add(int *a, int *b, int *c) {
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// CPU multiply
void mulVecMat(double *matrix, double *input, double *output, int matrix_size){
  for(int i = 0; i < matrix_size; i++){
    output[i] = 0;
    for(int j = 0; j < matrix_size; j++){
      output[i] += input[j] * matrix[i * matrix_size + j];
    }
  }
}


// CPU function to generate a vector of random integers
void random_ints (int *a, int n) {
    for (int i = 0; i < n; i++)
        a[i] = rand() % 10000; // random number between 0 and 9999
}

double* mulVecMat(double *input, double **matrix, int row, int col){
  double *output = new double[row];
  
  double *dev_output; cudaMalloc((void **) &dev_output, sizeof(double)*row);
  double **dev_matrix; cudaMalloc((void **) &dev_matrix, sizeof(double)*row*col);
  double *dev_input; cudaMalloc((void **) &dev_input, sizeof(double)*row);

  cudaMemcpy(dev_matrix, matrix, sizeof(double)*row*col, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_input, input, sizeof(double)*row, cudaMemcpyHostToDevice);

  unsigned int grid_rows = (row + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  //gpu_vecMatrix_mult<<<dimGrid, dimBlock>>>(dev_matrix, dev_input, dev_output, row, col);

  cudaMemcpy(output, dev_output, sizeof(double)*row, cudaMemcpyDeviceToHost);
  
  cudaFree(dev_output);
  cudaFree(dev_matrix);
  cudaFree(dev_input);

  return output;
}

double rand_init(double max_weight){
  double x = ((double) rand() / (RAND_MAX));
  return (x * 2 - 1) * max_weight;
}


int main(int argc, char** argv) {
  cerr << "Start Main\n";
  double max_weight = 1;
  int matrix_size = 5910;
  
  time_t startTime = time(nullptr);
  srand(startTime); // seed random

  cerr << "Create Matrices\n";
  // create input vector
  double *output;
  double *input;
  double *matrix;

  cerr << "Create Matrices on the gpu\n";
  cudaMallocManaged(&input, matrix_size*sizeof(double));
  cudaMallocManaged(&output, matrix_size*sizeof(double));
  cudaMallocManaged(&matrix, matrix_size*matrix_size*sizeof(double));


  //cerr << " [";
  for(int i = 0; i < matrix_size; i++){
    input[i] = rand_init(max_weight);
    //cerr << " " << input[i];
  }
  //cerr << " ]\n";

  //cerr << "\n";
  for(int i = 0; i < matrix_size; i++){
    //cerr << " [";
    for(int j = 0; j < matrix_size; j++){
      int idx = i * matrix_size + j;
      matrix[idx] = rand_init(max_weight);
      //cerr << " " << matrix[idx];
    }
    //cerr << " ]\n";
  }
  //cerr << "\n";

  cerr << "Run the Multiplication\n";
  // GTX 690 max threads per block 1024
  int deviceId = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceId);

  int blockSize = prop.maxThreadsPerBlock;
  int numBlocks = (matrix_size + blockSize - 1) / blockSize;

  using namespace std::chrono;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  gpu_vecMatrix_mult<<<numBlocks,blockSize>>>(matrix, input, output, matrix_size);
  cudaDeviceSynchronize();

  //mulVecMat(matrix, input, output, matrix_size);

  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  duration<double, std::milli> time_span = t2 - t1;

  std::cout << "It took me " << time_span.count() << " milliseconds.";
  std::cout << std::endl;

  //cerr << "Print output returned\n";
  //cerr << " [";
  //for(int i = 0; i < matrix_size; i++){
  //  cerr << " " << output[i];
  //}
  //cerr << " ]\n";

  cerr << "Clear the memory on the GPU\n";
  cudaFree(input); cudaFree(output); cudaFree(matrix);

  cerr << "Ending Main\n";
}


int old_main(void) {
  int *a, *b, *c;              // host copies of a, b, c
  int *d_a, *d_b, *d_c;      // device copies of a, b, c
  int size = N * sizeof(int);
              
  // Allocate space for device copies of a, b, c
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

