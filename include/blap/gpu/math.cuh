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

#ifndef blap_gpu_math_cuh_
#define blap_gpu_math_cuh_

#include <stdio.h>

namespace blap {
  namespace gpu {
    __global__ void matrixInit(double *matrix, unsigned int row, unsigned int col, double value);

    __global__ void matrixNorm(
        double *matrix_norm, double *matrix,
        unsigned int row, unsigned int col);

    __global__ void vectorMatrixMultiply(
        double *matrix, double *input, double *output,
        unsigned int row, unsigned int col,
        bool bias_en, double bias );

    __global__ void updateWeightsDeltaAdam(
        double *weight_delta, double *prev_weight_delta,
        int rows, int cols,
        double *input_vector, double *delta,
        bool bias_enabled, double bias,
        double learning_rate, double lambda,
        double beta1, double beta2, double eta, int step,
        double *moment_first, double *moment_second );
    
    __global__ void updateWeightsDeltaSgd(
        double *weight_delta, double *prev_weight_delta,
        int rows, int cols,
        double *input_vector, double *delta,
        bool bias_enabled, double bias,
        double learning_rate, double lambda );
    
    __global__ void checkDelta(double *weight_delta, int rows, int cols);
    
    __global__ void updateMatrix(
        double *matrix,
        double *weight_delta, double *prev_weight_delta,
        int rows, int cols,
        double *momentum,
        double momentum_missmatch,
        double momentum_start,
        double momentum_max,
        double momentum_step,
        double momentum_rho,
        double learning_rate,
        double weight_decay );
   
    __global__ void batchNormMean(double *mean, double *inputs, unsigned int input_size, unsigned int num_inputs);
    __global__ void batchNormVariance(double *variance, double *mean, double *inputs, unsigned int input_size, unsigned int num_inputs);
    __global__ void batchNormScale(double * outputs, double *inputs, double *mean, double *variance, double *gamma, double *batch_beta, unsigned int input_size, unsigned int num_inputs);

    __global__ void clearVector(double *vector, unsigned int size);
    
    __global__ void updateErrorUp(double *prev_error, double *error, double *matrix, unsigned int input_size, unsigned int output_size);
    
    __global__ void diffVector(double *output, double *vector_a, double *vector_b, unsigned int size);
    
    __global__ void pointMultiplyVector(double *output, double *vector_a, double *vector_b, unsigned int size);
    
    __global__ void printVector(double *vector, unsigned int size);
  }
}

#endif  // blap_gpu_math_cuh_


