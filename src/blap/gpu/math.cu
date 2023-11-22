//  blap - deep learning library
//    "blap, sometimes it just hits you in the face."
//
//  Copyright (C) 2017  Saul Rosa http://www.megaframe.org/blap
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "blap/gpu/math.cuh"

__global__ void blap::gpu::matrixInit(double *weight, unsigned int row, unsigned int col, double value){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x;
  if(index >= row * col){ return; }
  weight[index] = value;
}

__global__ void blap::gpu::matrixNorm(
    double *matrix_norm, double *matrix,
    unsigned int row, unsigned int col){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x;

  if(index >= row){ return; }

  unsigned int col_range = col;

  double sum = 0;
  for(unsigned int j = 0; j < col_range; j++){
    sum += matrix[index * col + j] * matrix[index * col + j];
  }
  matrix_norm[index] = sqrt(sum);
}

__global__ void blap::gpu::vectorMatrixMultiply(
    double *weight, double *input, double *output,
    unsigned int row, unsigned int col,
    bool bias_en, double bias){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x;

  if(index >= row){ return; }

  unsigned int col_range = col;
  if(bias_en){ col_range--; }

  double sum = 0;
  for(unsigned int j = 0; j < col_range; j++){
    sum += input[j] * weight[index * col + j];
  }
  if(bias_en){
    sum += bias * weight[index * col + col_range];
  }
  output[index] = sum;
}

__global__ void blap::gpu::updateWeightsDeltaAdam(
    double *weight_delta, double *prev_weight_delta, 
    int rows, int cols,
    double *input_vector, double *delta,
    bool bias_enabled, double bias,
    double learning_rate, double lambda,
    double beta1, double beta2, double eta, int step,
    double *moment_first, double *moment_second
    ){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= rows * cols){ return; }
  prev_weight_delta[index] = weight_delta[index];
  int i = index / cols;
  int j = index % cols;
  double vector_value = 0;
  if((j == cols - 1) and bias_enabled){ vector_value = bias; }
  else{ vector_value = input_vector[j]; }

  //double l2_norm = 0;
  //if(lambda > 0){
  //  l2_norm = weight_delta[index];
    //for(unsigned int i = 0; i < rows; i++){
    //  l2_norm += weight_norm[i];
    //}
  //}

//  printf("Adam, index %i moment_first %f deta %f i %f vector_value %f\n", 
//      index,
//      moment_first[index],
//      delta[i], i, vector_value);
  moment_first[index] = beta1 * moment_first[index] + ( 1 - beta1 ) * delta[i] * vector_value;
  moment_second[index] = beta2 * moment_second[index] + ( 1 - beta2 ) * delta[i] * delta[i] * vector_value * vector_value;
  double moment_first_bias_correction = moment_first[index] / (1 - pow(beta1,(double)step));
  double moment_second_bias_correction = moment_second[index] / (1 - pow(beta2,(double)step));

//  printf("Adam, index %i moment_first %f moment_second %f moment_first_bias_correction %f moment_second_bias_correction %f vector_value %f i %i beta1 %f beta2 %f eta %f step %i\n",
//      index,
//      moment_first[index], moment_second[index], 
//      moment_first_bias_correction, moment_second_bias_correction, 
//      vector_value, i, beta1, beta2, eta, step
//      );

  weight_delta[index] = -learning_rate * moment_first_bias_correction / sqrt(moment_second_bias_correction + eta);
}


__global__ void blap::gpu::updateWeightsDeltaSgd(
    double *weight_delta, double *prev_weight_delta, 
    int rows, int cols, 
    double *input_vector, double *delta,
    bool bias_enabled, double bias,
    double learning_rate, double lambda
    ){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= rows * cols){ return; }
  prev_weight_delta[index] = weight_delta[index];
  int i = index / cols;
  int j = index % cols;
  double vector_value = 0;
  if((j == cols - 1) and bias_enabled){ vector_value = bias; }
  else{ vector_value = input_vector[j]; }

  //weight_delta[index] = vector_value * delta[i];

  double l2_norm = 0;
  if(lambda > 0){
    l2_norm = weight_delta[index];
    //for(unsigned int i = 0; i < rows; i++){
    //  l2_norm += weight_norm[i];
    //}
  }
  //double update = -learning_rate * vector_value * delta[i] + lambda * 0.5 * l2_norm;
  //printf("lambda %f l2_norm %f update %f weight[index] %f\n",lambda,l2_norm,update,weight[index]);

  weight_delta[index] = -learning_rate * vector_value * delta[i] - lambda * 0.5 * l2_norm;
}

__global__ void blap::gpu::checkDelta(double *weight_delta, int rows, int cols){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index > 0){ return; }

  double max_value = 0;
  for(unsigned int i = 0; i < rows; i++){
    for(unsigned int j = 0; j < cols; j++){
      if(weight_delta[i * cols + j] > max_value){ max_value = weight_delta[i * cols + j]; }
    }
  }
  if(max_value <= 0){
    printf("WARNING: No Update done!\n");
  }
}

__global__ void blap::gpu::updateMatrix(
    double *weight, double *weight_delta, double *prev_weight_delta,
    int rows, int cols,
    double *momentum,
    double momentum_missmatch,
    double momentum_start,
    double momentum_max,
    double momentum_step,
    double momentum_rho,
    double learning_rate,
    double weight_decay
    ){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= rows * cols){ return; }
  int i = index / cols;
  int j = index % cols;

  /*
  // dx = gradient(x)
  // vx = rho * v(x) + dx
  // x += learning_rate * v

  double dx = weight_delta[i * cols + j];
  double vx = momentum_rho *  momentum[i * cols + j] + dx;
  momentum[i * cols + j] = vx;
  weight[i * cols + j] = 0.97 * weight[i * cols + j];
  weight[i * cols + j] += -learning_rate * vx;
  */

  double momentum_add = 0;
  if( (weight_delta[i * cols + j] > 0 and prev_weight_delta[i * cols + j] > 0) or
      (weight_delta[i * cols + j] < 0 and prev_weight_delta[i * cols + j] < 0) ){
    momentum_add = momentum[i * cols + j] * prev_weight_delta[i * cols + j];
    if(momentum[i * cols + j] < momentum_max){ momentum[i * cols + j] += momentum_step; }
  }
  else{
    momentum[i * cols + j] = momentum_start;
    momentum_add = momentum_missmatch * prev_weight_delta[i * cols + j];
  }
  weight_delta[i * cols + j] += weight_delta[i * cols + j] + momentum_add;

  weight[i * cols + j] += weight_delta[i * cols + j] - weight_decay * weight[i * cols + j];
}

__global__ void batchNormMean(double *mean, double *inputs, unsigned int input_size, unsigned int num_inputs){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= input_size){ return; }

  double sum = 0;
  for(unsigned int set_index = 0; set_index < num_inputs; set_index++){
    sum += inputs[index + set_index * num_inputs];
  }
  mean[index] = sum / num_inputs;
}

__global__ void batchNormVariance(double * variance, double *mean, double *inputs, unsigned int input_size, unsigned int num_inputs){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= input_size){ return; }

  double sum = 0;
  for(unsigned int set_index = 0; set_index < num_inputs; set_index++){
    sum += ( (inputs[index + set_index * num_inputs] - mean[index]) * (inputs[index + set_index * num_inputs] - mean[index]) );
  }
  variance[index] = sum / num_inputs;
}

__global__ void batchNormScale(double *outputs, double *inputs, double *mean, double *variance, double *gamma, double *batch_beta, unsigned int input_size, unsigned int num_inputs){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= input_size){ return; }

  for(unsigned int set_index = 0; set_index < num_inputs; set_index++){
    double normalized = (inputs[index + set_index * num_inputs] - mean[index]) / sqrt(variance[index] + 10e-8);
    outputs[index + set_index * num_inputs] = gamma[index] * normalized + batch_beta[index];
  }
}

__global__ void blap::gpu::updateErrorUp(double *prev_error, double *error, double *weight, unsigned int input_size, unsigned int output_size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= input_size){ return; }
  int j = index;
  prev_error[j] = 0;
  for(unsigned int i = 0; i < output_size; i++){
    prev_error[j] += error[i] * weight[i * input_size + j]; 
  }
}

__global__ void blap::gpu::diffVector(double *output, double *vector_a, double *vector_b, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  output[index] = vector_a[index] - vector_b[index];
}

__global__ void blap::gpu::pointMultiplyVector(double *output, double *vector_a, double *vector_b, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  output[index] = vector_a[index] * vector_b[index];
}

__global__ void blap::gpu::printVector(double *vector, unsigned int size){
  printf("GPU Print Vector:\n");
  for(unsigned int i = 0; i < size; i++){
    printf("gpu i %d vector[i] %f\n",i,vector[i]);
  }
}

