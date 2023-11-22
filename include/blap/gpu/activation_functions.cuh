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

#ifndef blap_gpu_activation_functions_cuh_
#define blap_gpu_activation_functions_cuh_

namespace blap {
  namespace gpu {
    namespace activation {
      __global__ void softmax(double *output, double *input_vector, unsigned int size);;
      __global__ void softmax_derivative(double *output, double *input_vector, unsigned int size);
      
      __global__ void g_tanh(double *output, double *input_vector, unsigned int size);
      __global__ void g_tanh_derivative(double *output, double *input_vector, unsigned int size);
      
      __global__ void ReLU(double *output, double *input_vector, unsigned int size);
      __global__ void ReLU_derivative(double *output, double *input_vector, unsigned int size);
      
      __global__ void leakyReLU(double *output, double *input_vector, unsigned int size);
      __global__ void leakyReLU_derivative(double *output, double *input_vector, unsigned int size);
      
      __global__ void SELU(double *output, double *input_vector, unsigned int size);
      __global__ void SELU_derivative(double *output, double *input_vector, unsigned int size);
      
      __global__ void ResNet(double *output, double *input_vector, unsigned int size, double *resNet_fx);
      __global__ void ResNet_derivative(double *output, double *input_vector, unsigned int size);
      
      __global__ void ELU(double *output, double *input_vector, unsigned int size);
      __global__ void ELU_derivative(double *output, double *input_vector, unsigned int size);
      
      __global__ void linear(double *output, double *input_vector, unsigned int size);
      __global__ void linear_derivative(double *output, double *input_vector, unsigned int size);

      __global__ void sigmoid(double *output, double *input_vector, unsigned int size);
      __global__ void sigmoid_derivative(double *output, double *input_vector, unsigned int size);
    }
  }
}

#endif  // blap_gpu_activation_functions_cuh_


