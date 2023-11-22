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

#ifndef blap_Model_gpu_h_
#define blap_Model_gpu_h_

#define BLAP_MAJOR_VERSION 0
#define BLAP_MIN_VERSION 3

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <utility>
#include <sys/stat.h>
#include <math.h>
#include <random>

#include "blap/ActivationFunctions.h"
#include "blap/InitFunctions.h"
#include "blap/Utils.h"

#include "blap/gpu/math.cuh"
#include "blap/gpu/activation_functions.cuh"

#include "driver_types.h"

using namespace std;

namespace blap {

  struct Layer {
    Layer *next_layer = NULL;
    Layer *prev_layer = NULL;

    bool initialized = false;
    bool weights_initialized = false;

    unsigned int input_size = 0;
    unsigned int output_size = 0;
    string activiation_function;

    double bias;
    bool bias_enabled = false;

    /* Batch Norm Vectors */
    double *batch_mean;
    double *batch_variance;
    double *batch_gamma;
    double *batch_beta;
    double *batch_outputs;

    // TODO implement device compute
    double *d_batch_mean;
    double *d_batch_variance;
    double *d_batch_gamma;
    double *d_batch_beta;
    double *d_batch_outputs;

    bool batch_norm = false;

    /* Host Data */
    double *weight;
    
    double *weight_delta;
    double *prev_weight_delta;
    
    double *momentum;

    double *moment_first;
    double *moment_second;
    
    double *weight_norm;

    double *output_vector;
    double *output_activation;
    double *output_activation_derivative;
    
    double *error;
    double *delta;

    /* Device Data */
    double *d_weight;
    
    double *d_weight_delta;
    double *d_prev_weight_delta;
    
    double *d_momentum;

    double *d_moment_first;
    double *d_moment_second;

    double *d_weight_norm;
    
    double *d_output_vector;
    double *d_output_activation;
    double *d_output_activation_derivative;
    
    double *d_error;
    double *d_delta;

  };

  struct DataItem {
    double * input;
    double * output;
  };

  typedef double (*ActivationPointer)(double);

  class Model {
    private:
      double *mulVecMat(double *input, double *matrix, int row, int col);
      double *mulVecMat_cpu(double *input, double *matrix, int row, int col);
      void compute_activiation(
          double *output_activation,
          double *output_activation_derivative,
          string activiation_function,
          double *output_vector,
          unsigned int size,
          Layer *current_layer,
          double *d_sample_vector);
      void compute_batch_norm_mean(
          double *d_batch_mean,
          double *output_activation,
          unsigned int size,
          unsigned int mini_batch_size);
      void compute_batch_norm_variance(
          double *d_batch_variance,
          double *d_batch_mean,
          double *output_activation,
          unsigned int size,
          unsigned int mini_batch_size);
      void compute_batch_norm_scale_shift(
          double *d_batch_outputs,
          double *output_activation,
          double *d_batch_mean,
          double *d_batch_variance,
          double *d_batch_gamma,
          double *d_batch_beta,
          unsigned int size,
          unsigned int mini_batch_size);

      void mulVecMat_gpu(double *output, double *input, double *matrix, bool bias_en, double bias, unsigned int row, unsigned int col);
      bool compareVector(double *vec_a, double *vec_b, int row);
      void handleCudaCode(cudaError_t cudaErrorCode);
      void checkGpuStatus();
      unsigned int blockSize = 0;
      string log_file;
      bool log_file_set = false;
      void writeLogFile( 
          double test_reg_error, double test_class_error, 
          double train_reg_error, double train_class_error,
          unsigned int test_size, unsigned int train_size);
      void writeLogFile(
          double test_reg_error, double train_reg_error,
          unsigned int test_size, unsigned int train_size);
      void setupFunctionLinks();
      void setupGpuBlockSize();
      void build_layer(Layer *current_layer);
      void build_layer_weights(Layer *current_layer);
      bool training = true;

    public:
      int gpuId = 0;
      double min_weight = 0;
      double max_weight = 0.5;
      string weight_init_func = "rand";
      string grad_optimizer = "sgd"; // sgd, adam
      double learning_rate = 0.1;
      double weight_decay = 0; // or 1e-4
      double beta1 = 0.9;
      double moment_first_start = 0;
      double beta2 = 0.999;
      double moment_second_start = 0;
      double eta = 10e-8;
      double momentum_matching = 0.05;
      double momentum_missmatch = 0.5;
      double momentum_max = 2.5;
      double stop_threshold = 0.0001;
      double max_iterations = 3000;
      double momentum_step = 0.1;
      double momentum_start = 0.05;
      double momentum_rho = 0; // Good default 0.98
      double lambda = 0; // Good default 0.03
      unsigned int mini_batch_size = 1; // If  > 1 set mini_batch vectors are enabled
      bool pre_loaded_model = false;
      bool verbose = false;
      bool no_check = false; // Turn off Train/Test checking at epochs
      unsigned int epoch_interval = 1;
      unsigned int epoch = 0;
      time_t startTime = time(nullptr);

      map<string, ActivationPointer> activation_function_map;
      unsigned int model_input_size;
      Layer *front_layer;
      Layer *back_layer;

      Model();
      Model(unsigned int gpuId);
      ~Model();

      vector<DataItem> read_data(string data_file);
      void delete_data(vector<blap::DataItem> test_set);
      
      void setup_model(string model_file);
      void train(vector<DataItem> training_set, vector<DataItem> test_set);
      double *solve(double *input);
      void error(double &classification_error, double &regression_error, vector<DataItem> &test_set);
      double classification_error(vector<DataItem> &test_set);
      double regression_error(vector<DataItem> &test_set);
      void printModel();
      void save(string save_file);
      void modelCopyHostToDevice();
      void modelCopyDeviceToHost();
      void setLogFile(string set_log_file);
      void disableTraining();
  };

}

#endif  // blap_Model_gpu_h_


