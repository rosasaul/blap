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

#ifndef blap_Solvers_cuh_
#define blap_Solvers_cuh_

using namespace std;

namespace blap {
  namespace solvers {
    __global__ void softmax_gpu(double *output, double *input_vector, unsigned int size);
    __global__ void softmax_derivative_gpu(double *output, double *input_vector, unsigned int size);

    __global__ void tanh_gpu(double *output, double *input_vector, unsigned int size);
    __global__ void tanh_derivative_gpu(double *output, double *input_vector, unsigned int size);

    __global__ void ReLU_gpu(double *output, double *input_vector, unsigned int size);
    __global__ void ReLU_derivative_gpu(double *output, double *input_vector, unsigned int size);

    __global__ void leakyReLU_gpu(double *output, double *input_vector, unsigned int size);
    __global__ void leakyReLU_derivative_gpu(double *output, double *input_vector, unsigned int size);

    __global__ void SELU_gpu(double *output, double *input_vector, unsigned int size);
    __global__ void SELU_derivative_gpu(double *output, double *input_vector, unsigned int size);

    __global__ void ResNet_gpu(double *output, double *input_vector, unsigned int size, double *resNet_fx);
    __global__ void ResNet_derivative_gpu(double *output, double *input_vector, unsigned int size);

    __global__ void ELU_gpu(double *output, double *input_vector, unsigned int size);
    __global__ void ELU_derivative_gpu(double *output, double *input_vector, unsigned int size);

    __global__ void linear_gpu(double *output, double *input_vector, unsigned int size);
    __global__ void linear_derivative_gpu(double *output, double *input_vector, unsigned int size);

    __global__ void sigmoid_gpu(double *output, double *input_vector, unsigned int size);
    __global__ void sigmoid_derivative_gpu(double *output, double *input_vector, unsigned int size);
  }
}

#endif  // blap_Solvers_cuh_


