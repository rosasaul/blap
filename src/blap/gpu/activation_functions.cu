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

#include "blap/gpu/activation_functions.cuh"


__global__ void blap::gpu::activation::softmax(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  double max_val = input_vector[0];
  for(unsigned int i = 1; i < size; i++){
    if(max_val < input_vector[i]){ max_val = input_vector[i]; }
  }
  double sum_of_exp = 0;
  for(unsigned int i = 0; i < size; i++){
    sum_of_exp += exp(input_vector[i] - max_val);
  }
  output[index] = exp(input_vector[index] - max_val) / sum_of_exp;
}

__global__ void blap::gpu::activation::softmax_derivative(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  double max_val = input_vector[0];
  for(unsigned int i = 1; i < size; i++){
    if(max_val < input_vector[i]){ max_val = input_vector[i]; }
  }
  double sum_of_exp = 0;
  for(unsigned int i = 0; i < size; i++){
    sum_of_exp += exp(input_vector[i] - max_val);
  }
  double y_i = exp(input_vector[index] - max_val) / sum_of_exp;
  output[index] = y_i * (1 - y_i);
}

__global__ void blap::gpu::activation::g_tanh(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  output[index] = (tanh(input_vector[index]) + 1) / 2;
}

__global__ void blap::gpu::activation::g_tanh_derivative(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  output[index] = ((1 - (tanh(input_vector[index]) * tanh(input_vector[index])))/2);
}

__global__ void blap::gpu::activation::ReLU(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  if(input_vector[index] < 0){ output[index] = 0; }
  else{ output[index] = input_vector[index]; }
}

__global__ void blap::gpu::activation::ReLU_derivative(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  if(input_vector[index] < 0){ output[index] = 0; }
  else{ output[index] = 1; }
}

__global__ void blap::gpu::activation::leakyReLU(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  if(input_vector[index] < 0){ output[index] = 0.01 * input_vector[index]; }
  else{ output[index] = input_vector[index]; }
}

__global__ void blap::gpu::activation::leakyReLU_derivative(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  if(input_vector[index] < 0){ output[index] = 0.01; }
  else{ output[index] = 1; }
}

__global__ void blap::gpu::activation::SELU(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  double alpha = 1.6732632423543772848170429916717;
  double scale = 1.0507009873554804934193349852946;
  if(input_vector[index] > 0){ output[index] = scale * input_vector[index]; }
  else{ output[index] = scale * (alpha * exp(input_vector[index]) - alpha); }
}

__global__ void blap::gpu::activation::SELU_derivative(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  double alpha = 1.6732632423543772848170429916717;
  double scale = 1.0507009873554804934193349852946;
  if(input_vector[index] > 0){ output[index] = scale; }
  else{ output[index] = scale * alpha * exp(input_vector[index]); }
}


__global__ void blap::gpu::activation::ResNet(double *output, double *input_vector, unsigned int size, double *resNet_fx){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  if(input_vector[index] < 0){ output[index] = 0.01 * input_vector[index] + resNet_fx[index]; }
  else{ output[index] = input_vector[index] + resNet_fx[index]; }
}

__global__ void blap::gpu::activation::ResNet_derivative(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  if(input_vector[index] < 0){ output[index] = 1.01; }
  else{ output[index] = 2; }
}


__global__ void blap::gpu::activation::ELU(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  if(input_vector[index] < 0){ output[index] = 0.2 * (exp(input_vector[index]) - 1); }
  else{ output[index] = input_vector[index]; }
}

__global__ void blap::gpu::activation::ELU_derivative(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  if(input_vector[index] < 0){ output[index] = 0.2 * exp(input_vector[index]); }
  else{ output[index] = 1; }
}

__global__ void blap::gpu::activation::linear(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  output[index] = input_vector[index];
}

__global__ void blap::gpu::activation::linear_derivative(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  output[index] = 1;
}

__global__ void blap::gpu::activation::sigmoid(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  output[index] = 1 / ( 1 + exp(0 - input_vector[index]) );
}

__global__ void blap::gpu::activation::sigmoid_derivative(double *output, double *input_vector, unsigned int size){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size){ return; }
  double yx = 1 / ( 1 + exp(0 - input_vector[index]) );
  output[index] = yx * ( 1 - yx );
}


