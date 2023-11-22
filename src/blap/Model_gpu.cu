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

#include "blap/Model_gpu.h"
#include "blap/Utils.h"
#include "blap/gpu/math.cuh"
#include "blap/gpu/activation_functions.cuh"

using namespace std;
using namespace blap;

blap::Model::Model(){
  setupGpuBlockSize();
  setupFunctionLinks();
}

blap::Model::Model(unsigned int gpuId){
  this->gpuId = gpuId;
  this->setupGpuBlockSize();
  this->setupFunctionLinks();
}

void blap::Model::disableTraining(){
  training = false;
  mini_batch_size = 1; // Even if it's set in the model it should be 1 for non-training
}

void blap::Model::setupGpuBlockSize(){
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, gpuId);
  this->blockSize = prop.maxThreadsPerBlock;
}

void blap::Model::handleCudaCode(cudaError_t cudaErrorCode){
  if(cudaErrorCode != cudaSuccess){
    printf("Cuda Error %i: %s\n", cudaErrorCode, cudaGetErrorString(cudaErrorCode));
    exit(1);
  }
  return;
}

void blap::Model::checkGpuStatus(){
  cudaError_t cudaErrorCode = cudaGetLastError();
  if(cudaErrorCode != cudaSuccess){
    printf("Cuda Error: %s\n", cudaGetErrorString(cudaErrorCode));
    exit(1);
  }
}


void blap::Model::setupFunctionLinks(){
  // define activiation function links
  activation_function_map["sigmoid"] = sigmoid_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["sigmoid_derivative"] = sigmoid_derivative;
  
  activation_function_map["tanh"] = tanh_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["tanh_derivative"] = tanh_derivative;
 
  activation_function_map["ELU"] = ELU_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["ELU_derivative"] = ELU_derivative;

  activation_function_map["SELU"] = ReLU_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["SELU_derivative"] = ReLU_derivative;
  
  activation_function_map["ReLU"] = ReLU_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["ReLU_derivative"] = ReLU_derivative;
  
  activation_function_map["leakyReLU"] = ReLU_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["leakyReLU_derivative"] = ReLU_derivative; 
  
  activation_function_map["linear"] = linear_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["linear_derivative"] = linear_derivative; 
}

double *blap::Model::mulVecMat_cpu(double *input, double *weight, int row, int col){
  double *output = new double[row];
  for(int i = 0; i < row; i++){
    output[i] = 0;
    for(int j = 0; j < col; j++){
      output[i] += input[j] * weight[i * col + j];
    }   
  }
  return output;
}

bool Model::compareVector(double *vec_a, double *vec_b, int row){
  for(int i = 0; i < row; i++){
    double v = vec_a[i] - vec_b[i];
    if(-0.000001 > v or v > 0.000001){ return false; }
  }
  return true;
}

Model::~Model(){
  // Delete insantiated layers
  Layer *current_layer = front_layer;
  Layer *prev_layer;
  while(current_layer != NULL){
    free(current_layer->weight);
    free(current_layer->output_vector);
    free(current_layer->output_activation);
    free(current_layer->output_activation_derivative);
    if(training){
      free(current_layer->prev_weight_delta);
      free(current_layer->weight_delta);
      free(current_layer->momentum);
      if(grad_optimizer == "adam"){
        free(current_layer->moment_first);
        free(current_layer->moment_second);
      }
      free(current_layer->weight_norm);
      free(current_layer->delta);
      free(current_layer->error);
      if(mini_batch_size > 1){
        free(current_layer->batch_mean);
        free(current_layer->batch_variance);
        free(current_layer->batch_gamma);
        free(current_layer->batch_beta);
        free(current_layer->batch_outputs);
      }
    }

    cudaFree(current_layer->d_weight);
    cudaFree(current_layer->d_output_vector);
    cudaFree(current_layer->d_output_activation);
    cudaFree(current_layer->d_output_activation_derivative);

    if(training){
      cudaFree(current_layer->d_prev_weight_delta);
      cudaFree(current_layer->d_weight_delta);
      cudaFree(current_layer->d_momentum);
      if(grad_optimizer == "adam"){
        cudaFree(current_layer->d_moment_first);
        cudaFree(current_layer->d_moment_second);
      }
      cudaFree(current_layer->d_weight_norm);
      cudaFree(current_layer->d_delta);
      cudaFree(current_layer->d_error);
      if(mini_batch_size > 1){
        cudaFree(current_layer->d_batch_mean);
        cudaFree(current_layer->d_batch_variance);
        cudaFree(current_layer->d_batch_gamma);
        cudaFree(current_layer->d_batch_beta);
        cudaFree(current_layer->d_batch_outputs);
      }
    }

    // Update to next layer
    prev_layer = current_layer;
    current_layer = current_layer->next_layer;
    delete prev_layer;
  }
}

void Model::save(string save_file){
  ofstream outStream;
  outStream.exceptions ( ifstream::failbit | ifstream::badbit );
  // Check if file exists, die otherwise
  try {
    outStream.open(save_file);
  }
  catch (const ifstream::failure& e) {
    cerr << " \033[31mERROR:\033[0m opening/writing file '" << 
      save_file << 
      "' : " << strerror(errno) <<
      "\n";
    exit(1);
  }
  outStream.exceptions ( ifstream::badbit );
  outStream.precision(17);
  outStream<<"learning_rate=" << learning_rate << "\n";
  outStream<<"weight_decay=" << weight_decay << "\n";
  outStream<<"grad_optimizer=" << grad_optimizer << "\n";
  outStream<<"beta1=" << beta1 << "\n";
  outStream<<"beta2=" << beta2 << "\n";
  outStream<<"eta=" << eta << "\n";
  outStream<<"momentum_matching=" << momentum_matching << "\n";
  outStream<<"momentum_missmatch=" << momentum_missmatch << "\n";
  outStream<<"momentum_max=" << momentum_max << "\n";
  outStream<<"min_weight=" << min_weight << "\n";
  outStream<<"max_weight=" << max_weight << "\n";
  outStream<<"stop_threshold=" << stop_threshold << "\n";
  outStream<<"max_iterations=" << max_iterations << "\n";
  outStream<<"momentum_step=" << momentum_step << "\n";
  outStream<<"momentum_start=" << momentum_start << "\n";
  outStream<<"momentum_rho=" << momentum_rho << "\n";
  outStream<<"lambda=" << lambda << "\n";
  outStream<<"weight_init_func="<< weight_init_func <<"\n";
  outStream<<"input_size=" << model_input_size << "\n";
  outStream<<"epoch_interval=" << epoch_interval << "\n";
  outStream<<"epoch=" << epoch << "\n";

  // Copy the model back from the GPU to the Host
  modelCopyDeviceToHost();

  Layer *current_layer = front_layer;
  while(current_layer != NULL){
    outStream<<"layer:\n";
    outStream<<"  output_size="<< current_layer->output_size <<"\n";
    outStream<<"  activiation_function=" << current_layer->activiation_function << "\n";
    if(current_layer->bias_enabled){ outStream << "  bias=" << current_layer->bias << "\n"; }

    for(int i = 0; i < current_layer->output_size; i++){
      outStream<<"  [";
      for(int j = 0; j < current_layer->input_size; j++){
        outStream << "\t" << current_layer->weight[i * current_layer->input_size + j];
      }
      outStream<<"\n";
    }

    current_layer = current_layer->next_layer;
  }
  outStream.close();
}

void Model::modelCopyHostToDevice(){
  Layer *current_layer = front_layer;
  while(current_layer != NULL){
    cudaMemcpy(current_layer->d_weight, current_layer->weight, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyHostToDevice);
    if(training){
      cudaMemcpy(current_layer->d_weight_delta, current_layer->weight_delta, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(current_layer->d_prev_weight_delta, current_layer->prev_weight_delta, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(current_layer->d_momentum, current_layer->momentum, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyHostToDevice);
      if(grad_optimizer == "adam"){
        cudaMemcpy(current_layer->d_moment_first, current_layer->moment_first, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(current_layer->d_moment_second, current_layer->moment_second, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyHostToDevice);
      }
      cudaMemcpy(current_layer->d_weight_norm, current_layer->weight_norm, current_layer->output_size * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(current_layer->d_output_vector, current_layer->output_vector, current_layer->output_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(current_layer->d_output_activation, current_layer->output_activation, current_layer->output_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(current_layer->d_output_activation_derivative, current_layer->output_activation_derivative, current_layer->output_size * sizeof(double), cudaMemcpyHostToDevice);
    if(training){
      cudaMemcpy(current_layer->d_error, current_layer->error, current_layer->output_size * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(current_layer->d_delta, current_layer->delta, current_layer->output_size * sizeof(double), cudaMemcpyHostToDevice);
    }
    current_layer = current_layer->next_layer;
  }
}

void Model::modelCopyDeviceToHost(){
  Layer *current_layer = front_layer;
  while(current_layer != NULL){
    cudaMemcpy(current_layer->weight, current_layer->d_weight, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyDeviceToHost);
    if(training){
      cudaMemcpy(current_layer->weight_delta, current_layer->d_weight_delta, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(current_layer->prev_weight_delta, current_layer->d_prev_weight_delta, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(current_layer->momentum, current_layer->d_momentum, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyDeviceToHost);
      if(grad_optimizer == "adam"){
        cudaMemcpy(current_layer->moment_first, current_layer->d_moment_first, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(current_layer->moment_second, current_layer->d_moment_second, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyDeviceToHost);
      }
      cudaMemcpy(current_layer->weight_norm, current_layer->d_weight_norm, current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(current_layer->output_vector, current_layer->d_output_vector, current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(current_layer->output_activation, current_layer->d_output_activation, current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(current_layer->output_activation_derivative, current_layer->d_output_activation_derivative, current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
    if(training){
      cudaMemcpy(current_layer->error, current_layer->d_error, current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(current_layer->delta, current_layer->d_delta, current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
    }
    current_layer = current_layer->next_layer;
  }
}


void Model::setup_model(string model_file){
  ifstream inStream;
  inStream.exceptions ( ifstream::failbit | ifstream::badbit );

  smatch match_results;
  regex comment("#.*");
  regex re_grad_optimizer("^grad_optimizer=(\\S+)");
  regex re_learning_rate("^learning_rate=(\\S+)");
  regex re_weight_decay("^weight_decay=(\\S+)");
  regex re_beta1("^beta1=(\\S+)");
  regex re_beta2("^beta2=(\\S+)");
  regex re_eta("^eta=(\\S+)");
  regex re_momentum_matching("^momentum_matching=(\\S+)");
  regex re_momentum_missmatch("^momentum_missmatch=(\\S+)");
  regex re_momentum_max("^momentum_max=(\\S+)");
  regex re_min_weight("^min_weight=(\\S+)");
  regex re_max_weight("^max_weight=(\\S+)");
  regex re_stop_threshold("^stop_threshold=(\\S+)");
  regex re_max_iterations("^max_iterations=(\\S+)");
  regex re_momentum_step("^momentum_step=(\\S+)");
  regex re_momentum_start("^momentum_start=(\\S+)");
  regex re_momentum_rho("^momentum_rho=(\\S+)");
  regex re_lambda("^lambda=(\\S+)");
  regex re_mini_batch_size("^mini_batch_size=(\\d+)");
  regex re_weight_init_func("^weight_init_func=(\\S+)");
  regex re_input_size("^input_size=(\\S+)");
  regex re_layer_start("^layer:");
  regex re_output_size("^  output_size=(\\S+)");
  regex re_activiation_function("^  activiation_function=(\\S+)");
  regex re_bias("^  bias=(\\S+)");
  regex re_batch_norm("^  batch_norm=true");
  regex re_weight_line("^  \\["); // \\.-\t
  regex re_non_layer("^\\S+");
  regex re_epoch_interval("^epoch_interval=(\\d+)");
  regex re_epoch("^epoch=(\\d+)");

  // Check if file exists, die otherwise
  try {
    inStream.open(model_file);
  }
  catch (const ifstream::failure& e) {
    cerr << " \033[31mERROR:\033[0m opening/reading file '" << 
      model_file << 
      "' : " << strerror(errno) <<
      "\n";
    exit(1);
  }

  inStream.exceptions ( ifstream::badbit );

  front_layer = new Layer();
  Layer *current_layer = front_layer;


  // Counts layers for prints to screen
  int layer_count = 0;
  
  // Read in Model structure
  string line;
  bool redo_line = false;
  while( redo_line or getline(inStream,line) ){
    redo_line = false;
    line = regex_replace(line,comment,"");
    if(regex_search(line,match_results,re_grad_optimizer)){
      grad_optimizer = match_results[1].str();
    }
    else if(regex_search(line,match_results,re_learning_rate)){
      learning_rate = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_weight_decay)){
      weight_decay = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_beta1)){
      beta1 = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_beta2)){
      beta2 = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_eta)){
      eta = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_momentum_matching)){
      momentum_matching = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_momentum_missmatch)){
      momentum_missmatch = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_momentum_max)){
      momentum_max = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_min_weight)){
      min_weight = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_max_weight)){
      max_weight = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_stop_threshold)){
      stop_threshold = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_max_iterations)){
      max_iterations = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_momentum_step)){
      momentum_step = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_momentum_start)){
      momentum_start = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_momentum_rho)){
      momentum_rho = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_lambda)){
      lambda = stod(match_results[1].str().c_str());
    }
    else if(regex_search(line,match_results,re_mini_batch_size)){
      mini_batch_size = stoi(match_results[1].str().c_str());
    }
    else if(regex_search(line, match_results, re_weight_init_func)){
      weight_init_func = match_results[1].str();
    }
    else if(regex_search(line, match_results, re_epoch_interval)){
      epoch_interval = stoi(match_results[1].str().c_str());
    }
    else if(regex_search(line, match_results, re_epoch)){
      epoch = stol(match_results[1].str().c_str());
    }
    else if(regex_search(line, match_results, re_input_size)){
      model_input_size = stoi(match_results[1].str().c_str());
      current_layer->input_size = model_input_size;
    }
    else if(regex_search(line, match_results, re_layer_start)){
      unsigned int i = 0;
      while( getline(inStream,line) ){
        if(regex_search(line, match_results, re_output_size)){
          current_layer->output_size = stoi(match_results[1].str().c_str());
        }
        else if(regex_search(line, match_results, re_activiation_function)){
          current_layer->activiation_function = match_results[1].str();
        }
        else if(regex_search(line, match_results, re_bias)){
          current_layer->bias = stod(match_results[1].str().c_str());
          current_layer->bias_enabled = true;
          current_layer->input_size++;
        }
        else if(regex_search(line, match_results, re_batch_norm)){
          if(mini_batch_size < 2){
            cerr << " \033[31mERROR:\033[0m Mini Batch Size was not set\n";
            exit(1);
          }
          current_layer->batch_norm = true;
        }
        else if(regex_search(line, match_results, re_weight_line)){
          // Initialize weights
          build_layer(current_layer);
          current_layer->weights_initialized = true;

          line.erase(0,4); // Two spaces + [ + tab
          pre_loaded_model = true;

          unsigned int line_len = line.length() + 1;
          char * cstr = new char [line_len];
          strcpy(cstr, line.c_str());

          unsigned int begin_token = 0;
          unsigned int j = 0;

          for(unsigned int pos = 0; pos <= line_len; pos++){
            if(pos == line_len or cstr[pos] == '\t'){
              current_layer->weight[i * current_layer->input_size + j] = stod(&cstr[begin_token]);

              j++;
              if(j > current_layer->input_size){
                cerr << " \033[31mERROR:\033[0m Layer Weights Matrix count mismatch, "<<
                  "extra data found at layer " << layer_count << " row "<< i << " column " << j << "\n";
                exit(1);
              }
              begin_token = pos + 1;
            }
          }
          i++;
        }
        else if(
            regex_search(line, match_results, re_layer_start) or
            regex_search(line, match_results, re_non_layer)
            ){

          // Setup Matrixes if not already done
          build_layer(current_layer);
          build_layer_weights(current_layer);

          if( regex_search(line, match_results, re_layer_start) ){
            // Prepare for next Layer
            
            // Create new layer
            current_layer->next_layer = new Layer();
            
            // connect the layers
            current_layer->next_layer->prev_layer = current_layer;
            
            // switch to new layer
            current_layer = current_layer->next_layer;

            // Get input size from previous layer
            current_layer->input_size = current_layer->prev_layer->output_size;

            // Update layer count
            layer_count++;
            if(verbose){
              cerr << "  Layer " << layer_count << " Setup : " << 
                current_layer->prev_layer->activiation_function << " : " <<
                current_layer->prev_layer->input_size <<"x"<< current_layer->prev_layer->output_size << 
                "\n";
            }

            i = 0; // reset i for next layer
            continue;
          }
          redo_line = true;
          break;
        }
      }
      layer_count++;
      if(verbose){
        cerr << "  Layer " << layer_count << " Setup : " << 
          current_layer->activiation_function << " : " <<
          current_layer->input_size <<"x"<< current_layer->output_size << 
          "\n";
      }


      // Catch last layer in case model file ends on layer with no weights defined
      build_layer(current_layer);
      build_layer_weights(current_layer);
    }
  }

  inStream.close();

  back_layer = current_layer;

  // Get things onto the GPU
  modelCopyHostToDevice();
}

void Model::build_layer_weights(Layer *current_layer){
  if(!current_layer->initialized){
    cerr << " \033[31mERROR:\033[0m Layer not initialized before initializing weights\n";
    exit(1);
  }
  if(current_layer->weights_initialized){ return; }
  current_layer->weights_initialized = true;

  if(training){
    // used for Kaiming distribution
    default_random_engine generator;
    normal_distribution<double> distribution(0.0,1.0);

    // Update weight if no vector lines were specified
    for(unsigned int batch = 0; batch < mini_batch_size; batch++){
      for(unsigned int i = 0; i < current_layer->output_size; i++){
        for(unsigned int j = 0; j < current_layer->input_size; j++){
          unsigned int cord = i * current_layer->input_size + j + batch * current_layer->input_size * current_layer->output_size;

          if(weight_init_func == "rand"){
            current_layer->weight[cord] = rand_init(min_weight,max_weight);
          }
          else if(weight_init_func == "ones"){
            current_layer->weight[cord] = max_weight;
          }
          else if(weight_init_func == "kaiming"){
            current_layer->weight[cord] = distribution(generator) / sqrt((double)current_layer->input_size / 2);
          }
          else{
            cerr << " \033[31mERROR:\033[0m Uknown Weight initalization funnction defined '" << weight_init_func << "'\n";
            exit(1);
          }
        }
      }
    }
  }
  else{
    cerr << " \033[31mERROR:\033[0m Not in Training, weights not initilized\n";
    exit(1);
  }
}

void Model::build_layer(Layer *current_layer){
  if(current_layer->initialized){ return; }
  current_layer->initialized = true;

  /* Host Data */
  current_layer->weight = (double*)malloc(current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double));
  if(training){
    current_layer->weight_delta = (double*)malloc(current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double));
    current_layer->prev_weight_delta = (double*)malloc(current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double));
    current_layer->momentum = (double*)malloc(current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double));
    if(grad_optimizer == "adam"){
      current_layer->moment_first = (double*)malloc(current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double));
      current_layer->moment_second = (double*)malloc(current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double));
    }
    current_layer->weight_norm = (double*)malloc(current_layer->output_size * mini_batch_size * sizeof(double));

    if(mini_batch_size > 1){
      current_layer->batch_mean = (double*)malloc(current_layer->output_size * sizeof(double));
      current_layer->batch_variance = (double*)malloc(current_layer->output_size * sizeof(double));
      current_layer->batch_gamma = (double*)malloc(current_layer->output_size * sizeof(double));
      current_layer->batch_beta = (double*)malloc(current_layer->output_size * sizeof(double));
      current_layer->batch_outputs = (double*)malloc(current_layer->output_size * mini_batch_size * sizeof(double));
    }
  }
  current_layer->output_vector = (double*)malloc(current_layer->output_size * mini_batch_size * sizeof(double));
  current_layer->output_activation = (double*)malloc(current_layer->output_size * mini_batch_size * sizeof(double));
  current_layer->output_activation_derivative = (double*)malloc(current_layer->output_size * mini_batch_size * sizeof(double));
  if(training){
    current_layer->error = (double*)malloc(current_layer->output_size * mini_batch_size * sizeof(double));
    current_layer->delta = (double*)malloc(current_layer->output_size * mini_batch_size * sizeof(double));
  }

  /* Device Data */
  handleCudaCode(cudaMalloc(&current_layer->d_weight, current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double)));
  if(training){
    handleCudaCode(cudaMalloc(&current_layer->d_weight_delta, current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double)));
    handleCudaCode(cudaMalloc(&current_layer->d_prev_weight_delta, current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double)));
    handleCudaCode(cudaMalloc(&current_layer->d_momentum, current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double)));
    if(grad_optimizer == "adam"){
      handleCudaCode(cudaMalloc(&current_layer->d_moment_first, current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double)));
      handleCudaCode(cudaMalloc(&current_layer->d_moment_second, current_layer->output_size * current_layer->input_size * mini_batch_size * sizeof(double)));
    }
    handleCudaCode(cudaMalloc(&current_layer->d_weight_norm, current_layer->output_size * mini_batch_size * sizeof(double)));

    if(mini_batch_size > 1){
      handleCudaCode(cudaMalloc(&current_layer->d_batch_mean, current_layer->output_size * sizeof(double)));
      handleCudaCode(cudaMalloc(&current_layer->d_batch_variance, current_layer->output_size * sizeof(double)));
      handleCudaCode(cudaMalloc(&current_layer->d_batch_gamma, current_layer->output_size * sizeof(double)));
      handleCudaCode(cudaMalloc(&current_layer->d_batch_beta, current_layer->output_size * sizeof(double)));
      handleCudaCode(cudaMalloc(&current_layer->d_batch_outputs, current_layer->output_size * mini_batch_size * sizeof(double)));
    }
  }
  handleCudaCode(cudaMalloc(&current_layer->d_output_vector, current_layer->output_size * mini_batch_size * sizeof(double)));
  handleCudaCode(cudaMalloc(&current_layer->d_output_activation, current_layer->output_size * mini_batch_size * sizeof(double)));
  handleCudaCode(cudaMalloc(&current_layer->d_output_activation_derivative, current_layer->output_size * mini_batch_size * sizeof(double)));
  if(training){
    handleCudaCode(cudaMalloc(&current_layer->d_error, current_layer->output_size * mini_batch_size * sizeof(double)));
    handleCudaCode(cudaMalloc(&current_layer->d_delta, current_layer->output_size * mini_batch_size * sizeof(double)));

    // Update weight if no vector lines were specified
    for(unsigned int batch = 0; batch < mini_batch_size; batch++){
      for(unsigned int i = 0; i < current_layer->output_size; i++){
        for(unsigned int j = 0; j < current_layer->input_size; j++){
          unsigned int cord = i * current_layer->input_size + j + batch * current_layer->input_size * current_layer->output_size;
          current_layer->prev_weight_delta[cord] = 0;
          current_layer->weight_delta[cord] = 0;
          current_layer->momentum[cord] = momentum_start;
        }
      }
      if(grad_optimizer == "adam"){
        for(unsigned int i = 0; i < current_layer->output_size; i++){
          for(unsigned int j = 0; j < current_layer->input_size; j++){
            unsigned int cord = i * current_layer->input_size + j + batch * current_layer->input_size * current_layer->output_size;
            current_layer->moment_first[cord] = moment_first_start;
            current_layer->moment_second[cord] = moment_second_start;
          }
        }
      }
    }
  }
}

vector<DataItem> Model::read_data(string data_file){
  ifstream inStream;
  inStream.exceptions ( ifstream::failbit | ifstream::badbit );

  // Check if file exists, die otherwise
  try {
    inStream.open(data_file);
  }
  catch (const ifstream::failure& e) {
    cerr << " \033[31mERROR:\033[0m opening/reading file '" << 
      data_file << 
      "' : " << strerror(errno) <<
      "\n";
    exit(1);
  }
  inStream.exceptions ( ifstream::badbit );

  string line;
  vector<DataItem> dataset;

  //Read in data
  string delimiter = "\t";
  unsigned int max_tokens = model_input_size + back_layer->output_size;
  unsigned int line_num = 0;
  if(verbose){ fprintf(stderr, "line %d", line_num); }
  while( getline(inStream, line) ){
    DataItem data;
    data.input = new double[model_input_size];
    data.output = new double[back_layer->output_size];

    
    unsigned int line_len = line.length() + 1;
    char * cstr = new char [line_len];
    strcpy(cstr, line.c_str());

    int token_idx = 0;
    unsigned int begin_token = 0;
    double token;
    for(unsigned int pos = 0; pos <= line_len; pos++){
      if(pos == line_len or cstr[pos] == '\t'){
        token = stod(&cstr[begin_token]);

        if(token_idx >= max_tokens){
          cerr << " \033[31mERROR:\033[0m Line "<<line_num<<": More vectors in row than expected, max input " << max_tokens << "\n";
          exit(1);
        }
        if(token_idx < model_input_size){
          data.input[token_idx] = token;
        }
        else{
          data.output[token_idx - model_input_size] = token;
        }

        token_idx++;
        begin_token = pos + 1;
      }
    }

    dataset.push_back(data);
    line_num++;
    if(verbose){ fprintf(stderr, "\rline %d", line_num); }
  }
  if(verbose){ fprintf(stderr, "\n"); }

  inStream.close();

  return dataset;
}

void Model::delete_data(vector<blap::DataItem> test_set){
  for(unsigned int i = 0; i < test_set.size(); i++){
    delete[] test_set.at(i).input;
    delete[] test_set.at(i).output;
  }
}

void Model::printModel(){
  Layer *current_layer = front_layer;

  while(current_layer != NULL){
    blap::printMatrix(current_layer->output_size, current_layer->input_size, current_layer->weight);
    current_layer = current_layer->next_layer;
  }
}

double *Model::solve(double *input){
  double *sample_vector = (double*)malloc(this->front_layer->input_size * sizeof(double));

  for(unsigned int i = 0; i < model_input_size; i++){
    sample_vector[i] = input[i];
  }
  if(front_layer->bias_enabled){
    sample_vector[model_input_size] = front_layer->bias;
  }

  /* Device Data */
  double *d_sample_vector;
  handleCudaCode(cudaMalloc(&d_sample_vector, front_layer->input_size * sizeof(double)));
  // Copy over data
  handleCudaCode(cudaMemcpy(d_sample_vector, sample_vector, front_layer->input_size * sizeof(double), cudaMemcpyHostToDevice));

  free(sample_vector);

  // Init the layer
  Layer *current_layer = front_layer;

  while(!blap::trm and current_layer != NULL){
    double *input_vector;
    if(current_layer->prev_layer == NULL){ input_vector = d_sample_vector; }
    else{ input_vector = current_layer->prev_layer->d_output_activation; }

    // compute weighted weight multiplication
    mulVecMat_gpu(
        current_layer->d_output_vector,
        input_vector,
        current_layer->d_weight,
        current_layer->bias_enabled,
        current_layer->bias,
        current_layer->output_size,
        current_layer->input_size);
    if(blap::trm){ break; }

    // Compute activation / activation derivative
    compute_activiation(
        current_layer->d_output_activation,
        current_layer->d_output_activation_derivative,
        current_layer->activiation_function,
        current_layer->d_output_vector,
        current_layer->output_size,
        current_layer,
        d_sample_vector
        );

    // Exit if at end of chain
    if(current_layer->next_layer == NULL){ break; }

    // Update to next layer
    current_layer = current_layer->next_layer;
  }

  handleCudaCode(
      cudaMemcpy(
        current_layer->output_activation,
        current_layer->d_output_activation,
        current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost
        )
      );
  cudaFree(d_sample_vector);
  return current_layer->output_activation;
}

void Model::mulVecMat_gpu(double *output, double *input, double *weight, bool bias_en, double bias, unsigned int row, unsigned int col){
  int numBlocks = (row + blockSize - 1) / blockSize;
  blap::gpu::vectorMatrixMultiply<<<numBlocks,blockSize>>>(weight, input, output, row, col, bias_en, bias);
  cudaDeviceSynchronize();
  checkGpuStatus();
  return;
}

double *Model::mulVecMat(double *input, double *weight, int row, int col){
  double *d_output; double *d_input;
  handleCudaCode(cudaMallocManaged(&d_input, col*sizeof(double)));
  handleCudaCode(cudaMallocManaged(&d_output, row*sizeof(double)));

  // Copy in data
  for(int j = 0; j < col; j++){
    d_input[j] = input[j];
  }

  // Run the function
  int numBlocks = (row + blockSize - 1) / blockSize;
  blap::gpu::vectorMatrixMultiply<<<numBlocks,blockSize>>>(weight, d_input, d_output, row, col, false, 0);
  cudaDeviceSynchronize();
  checkGpuStatus();

  // Copy back data
  double *output = new double[row];
  for(int i = 0; i < row; i++){ output[i] = d_output[i]; }

  // Free up memory
  cudaFree(d_output); cudaFree(d_input); 

  return output;
}

void Model::compute_batch_norm_mean(
    double *d_batch_mean,
    double *output_activation,
    unsigned int size,
    unsigned int mini_batch_size
    ){
  int numBlocks = (size + blockSize - 1) / blockSize;
  // TODO Complete function
  // blap::gpu::batchNormMean<<<numBlocks,blockSize>>>(d_batch_mean, output_activation, size, mini_batch_size);
}

void Model::compute_batch_norm_variance(
    double *d_batch_variance,
    double *d_batch_mean,
    double *output_activation,
    unsigned int size,
    unsigned int mini_batch_size
    ){
  int numBlocks = (size + blockSize - 1) / blockSize;
  // TODO Complete function
  // blap::gpu::batchNormVariance<<<numBlocks,blockSize>>>(d_batch_variance, d_batch_mean, output_activation, size, mini_batch_size);
}

void Model::compute_batch_norm_scale_shift(
    double *d_batch_outputs,
    double *output_activation,
    double *d_batch_mean,
    double *d_batch_variance,
    double *d_batch_gamma,
    double *d_batch_beta,
    unsigned int size,
    unsigned int mini_batch_size){
  int numBlocks = (size + blockSize - 1) / blockSize;
  // TODO Complete function
  // blap::gpu::batchNormScale<<<numBlocks,blockSize>>>(d_batch_outputs, output_activation, d_batch_mean, d_batch_variance, d_batch_gamma, d_batch_beta, size, mini_batch_size);
}

void Model::compute_activiation(
    double *output_activation,
    double *output_activation_derivative,
    string activiation_function,
    double *input_vector,
    unsigned int size,
    Layer *current_layer,
    double *d_sample_vector
    ){ 
  int numBlocks = (size + blockSize - 1) / blockSize;
  if(activiation_function.compare("sigmoid") == 0){
    blap::gpu::activation::sigmoid<<<numBlocks,blockSize>>>(output_activation,input_vector,size);
    blap::gpu::activation::sigmoid_derivative<<<numBlocks,blockSize>>>(output_activation_derivative,input_vector,size);
  }
  else if(activiation_function.compare("softmax") == 0){
    blap::gpu::activation::softmax<<<numBlocks,blockSize>>>(output_activation,input_vector,size);
    blap::gpu::activation::softmax_derivative<<<numBlocks,blockSize>>>(output_activation_derivative,input_vector,size);
  }
  else if(activiation_function.compare("tanh") == 0){
    blap::gpu::activation::g_tanh<<<numBlocks,blockSize>>>(output_activation,input_vector,size);
    blap::gpu::activation::g_tanh_derivative<<<numBlocks,blockSize>>>(output_activation_derivative,input_vector,size);
  }
  else if(activiation_function.compare("ReLU") == 0){
    blap::gpu::activation::ReLU<<<numBlocks,blockSize>>>(output_activation,input_vector,size);
    blap::gpu::activation::ReLU_derivative<<<numBlocks,blockSize>>>(output_activation_derivative,input_vector,size);
  }
  else if(activiation_function.compare("ELU") == 0){
    blap::gpu::activation::ELU<<<numBlocks,blockSize>>>(output_activation,input_vector,size);
    blap::gpu::activation::ELU_derivative<<<numBlocks,blockSize>>>(output_activation_derivative,input_vector,size);
  }
  else if(activiation_function.compare("SELU") == 0){
    blap::gpu::activation::SELU<<<numBlocks,blockSize>>>(output_activation,input_vector,size);
    blap::gpu::activation::SELU_derivative<<<numBlocks,blockSize>>>(output_activation_derivative,input_vector,size);
  }
  else if(activiation_function.compare("leakyReLU") == 0){
    blap::gpu::activation::leakyReLU<<<numBlocks,blockSize>>>(output_activation,input_vector,size);
    blap::gpu::activation::leakyReLU_derivative<<<numBlocks,blockSize>>>(output_activation_derivative,input_vector,size);
  }
  else if(activiation_function.compare("linear") == 0){
    blap::gpu::activation::linear<<<numBlocks,blockSize>>>(output_activation,input_vector,size);
    blap::gpu::activation::linear_derivative<<<numBlocks,blockSize>>>(output_activation_derivative,input_vector,size);
  }
  else if(activiation_function.compare("ResNet") == 0){
    double *resNet_fx;
    if(current_layer->prev_layer->prev_layer == NULL){ resNet_fx = d_sample_vector; }
    else{ resNet_fx = current_layer->prev_layer->prev_layer->d_output_activation; }
    blap::gpu::activation::ResNet<<<numBlocks,blockSize>>>(output_activation,input_vector,size,resNet_fx);
    blap::gpu::activation::ResNet_derivative<<<numBlocks,blockSize>>>(output_activation_derivative,input_vector,size);
  }
  else{
    cerr << "ERROR: Unmapped activation function\n";
    exit(1);
  }
  cudaDeviceSynchronize();
  checkGpuStatus();
}

void Model::train(vector<DataItem> train_set, vector<DataItem> test_set){
  if(!training){
    cerr << "ERROR: Training was disabled\n";
    exit(1);
  }
  if(mini_batch_size < 1){
    cerr << "ERROR: mini_batch_size must be at least 1, was set to " << mini_batch_size << "\n";
    exit(1);
  }

  unsigned int num_train_set = train_set.size();
  unsigned int num_test_set = test_set.size();
  
  vector<int> sample_indexs;
  for(unsigned int i = 0; i < train_set.size(); i++){ sample_indexs.push_back(i); }

  /* Host Data */
  double *sample_vector = (double*)malloc(this->front_layer->input_size * mini_batch_size * sizeof(double));
  /* Device Data */
  double *d_sample_vector;
  handleCudaCode(cudaMalloc(&d_sample_vector, this->front_layer->input_size * mini_batch_size * sizeof(double)));

  /* Host Data */
  double *sample_output = (double*)malloc(back_layer->output_size * mini_batch_size * sizeof(double));
  /* Device Data */
  double *d_sample_output;
  handleCudaCode(cudaMalloc(&d_sample_output, back_layer->output_size * mini_batch_size * sizeof(double)));

  startTime = time(nullptr); // Update to latest

  unsigned int stop_epoch = max_iterations + epoch;

  for(; epoch <= stop_epoch; epoch++){
    if(blap::trm){ break; }

    // Check error
    if(epoch % epoch_interval == 0 and !no_check){
      if(verbose){
        fprintf(stderr, "[%s] Epoch %6d", blap::formatted(time(nullptr)).c_str(), epoch);
      }

      double test_reg_error = 1;
      double test_class_error = 1;
      double train_reg_error = 1;
      double train_class_error = 1;

      error(test_reg_error, test_class_error, test_set);
      error(train_reg_error, train_class_error, train_set);

      if(verbose){
        fprintf(stderr,
          " : Errors => Test Reg. %8.6f%% Class. %8.6f%% : Train Reg. %8.6f%% Class. %8.6f%%\n",
          test_reg_error * 100, test_class_error * 100,
          train_reg_error * 100, train_class_error * 100
        );
      }
      if(log_file_set){
        writeLogFile(
            test_reg_error, test_class_error, 
            train_reg_error, train_class_error, 
            num_test_set, num_train_set);
      }
      if(test_reg_error < stop_threshold){ break; }
    }
    else if(verbose){
      fprintf(stderr, "[%s] Epoch %6d\n", blap::formatted(time(nullptr)).c_str(), epoch );
    }

    // Shuffle up the training data
    random_shuffle(sample_indexs.begin(), sample_indexs.end());
    
    // Loop through each training item once per epoch
    for(unsigned int select_idx = 0; select_idx < num_train_set; select_idx += mini_batch_size){
      if(blap::trm){ break; }

      /* Cuda Profiler */
      // cudaProfilerStart();
      /* Cuda Profiler */

      for(unsigned int batch = 0; batch < mini_batch_size; batch++){
        unsigned int sample_index = sample_indexs[select_idx + batch];
        
        // Setup the sample input vector
        for(unsigned int i = 0; i < model_input_size; i++){
          sample_vector[i + batch * model_input_size] = train_set.at(sample_index).input[i];
        }
  
        // Setup the output vector
        for(unsigned int i = 0; i < back_layer->output_size; i++){
          sample_output[i + batch * back_layer->output_size] = train_set.at(sample_index).output[i];
        }
      }
      cudaMemcpy(d_sample_vector, sample_vector, model_input_size * mini_batch_size * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_sample_output, sample_output, back_layer->output_size * mini_batch_size * sizeof(double), cudaMemcpyHostToDevice);
  
      Layer *current_layer = this->front_layer;

      if(debug){
        cout << "Train vector : " << sample_indexs[select_idx] << " : ";
        printVector(current_layer->input_size, sample_vector);
      }

      ////////////////  Forward Pass ////////////////
      while(!blap::trm and current_layer != NULL){

        // Mini-Batch loop
        for(unsigned int batch = 0; batch < mini_batch_size; batch++){

          double *input_vector;
          if(current_layer->prev_layer == NULL){ input_vector = d_sample_vector; }
          else if(current_layer->prev_layer->batch_norm){ input_vector = current_layer->prev_layer->d_batch_outputs; }
          else{ input_vector = current_layer->prev_layer->d_output_activation; }
  
          mulVecMat_gpu(
              current_layer->d_output_vector + batch * current_layer->output_size,
              input_vector + batch * current_layer->input_size,
              current_layer->d_weight + batch * current_layer->input_size * current_layer->output_size,
              current_layer->bias_enabled,
              current_layer->bias,
              current_layer->output_size, 
              current_layer->input_size);
          if(blap::trm){ break; }
                
          if(debug){
            cudaMemcpy(current_layer->output_vector, 
                current_layer->d_output_vector, 
                current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
            cout << "output: ";
            printVector(current_layer->output_size, current_layer->output_vector);
          }
    
          // Compute activation / activation derivative
          compute_activiation(
              current_layer->d_output_activation + batch * current_layer->output_size,
              current_layer->d_output_activation_derivative + batch * current_layer->output_size,
              current_layer->activiation_function,
              current_layer->d_output_vector + batch * current_layer->output_size,
              current_layer->output_size,
              current_layer,
              d_sample_vector + batch * current_layer->input_size
              );
    
          if(debug){
            cudaMemcpy(current_layer->output_activation, 
                current_layer->d_output_activation, 
                current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
            cout << "output_activation: ";
            printVector(current_layer->output_size, current_layer->output_activation);
    
            cudaMemcpy(current_layer->output_activation_derivative, 
                current_layer->d_output_activation_derivative, 
                current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
            cout << "output_activation_derivative: ";
            printVector(current_layer->output_size, current_layer->output_activation_derivative);
          }
        }
  
        if(current_layer->batch_norm){ // done on a per dimention basis

          // TODO Compute batchnorm here
          // Compute mean
          compute_batch_norm_mean(
              current_layer->d_batch_mean,
              current_layer->d_output_activation,
              current_layer->output_size,
              mini_batch_size);
          // Compute sigma
          compute_batch_norm_variance(
              current_layer->d_batch_variance,
              current_layer->d_batch_mean,
              current_layer->d_output_activation,
              current_layer->output_size,
              mini_batch_size);
          // normalize + scale and shift
          compute_batch_norm_scale_shift(
              current_layer->d_batch_outputs,
              current_layer->d_output_activation,
              current_layer->d_batch_mean,
              current_layer->d_batch_variance,
              current_layer->d_batch_gamma,
              current_layer->d_batch_beta,
              current_layer->output_size,
              mini_batch_size);
        }
        
        // Exit if at end of chain
        if(current_layer->next_layer == NULL){ break; }
  
        // Update to next layer
        current_layer = current_layer->next_layer;
      }

      if(debug){
        cudaMemcpy(current_layer->output_activation, 
            current_layer->d_output_activation, 
            current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
        cout << "output_activation: ";
        printVector(current_layer->output_size, current_layer->output_activation);
      }
      
      //////////////// Back Propegation ////////////////
  
      // TODO run this for every mini_batch

      // Get error at end of chain
      double *final_output_vector;
      if(current_layer->batch_norm){ final_output_vector = current_layer->d_batch_outputs; }
      else{ final_output_vector = current_layer->d_output_activation; }

      unsigned int numBlocks = ((current_layer->output_size) + blockSize - 1) / blockSize;

      // Mini-Batch loop
      for(unsigned int batch = 0; batch < mini_batch_size; batch++){
        blap::gpu::diffVector<<<numBlocks,blockSize>>>(
            current_layer->d_error + batch * current_layer->output_size,
            final_output_vector + batch * current_layer->output_size,
            d_sample_output + batch * current_layer->output_size, 
            current_layer->output_size);
      }
      cudaDeviceSynchronize();
      checkGpuStatus();
  
      while(!blap::trm and current_layer != NULL){
  
        // Solve for delta
        unsigned int numBlocks = ((current_layer->output_size) + blockSize - 1) / blockSize;
        blap::gpu::pointMultiplyVector<<<numBlocks,blockSize>>>(
            current_layer->d_delta, 
            current_layer->d_error, 
            current_layer->d_output_activation_derivative, 
            current_layer->output_size);
        cudaDeviceSynchronize();
        checkGpuStatus();
        if(blap::trm){ break; }
  
        if(debug){
          cudaMemcpy(current_layer->error, current_layer->d_error, current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
          cout << "error: ";
          printVector(current_layer->output_size, current_layer->error);
  
          cudaMemcpy(current_layer->delta, current_layer->d_delta, current_layer->output_size * sizeof(double), cudaMemcpyDeviceToHost);
          cout << "Delta: ";
          printVector(current_layer->output_size, current_layer->delta);
        }
  
        double *input_vector;
        if(current_layer->prev_layer == NULL){
          input_vector = d_sample_vector;
        }
        else{
          input_vector = current_layer->prev_layer->d_output_activation;
        }

        // L2 Regularization, compute the weight column wise norm
        if(lambda > 0){
          int numBlocksNorm = (current_layer->output_size + blockSize - 1) / blockSize;
          blap::gpu::matrixNorm<<<numBlocksNorm,blockSize>>>(current_layer->d_weight_norm, current_layer->d_weight,
              current_layer->output_size, current_layer->input_size);
          cudaDeviceSynchronize();
          checkGpuStatus();
        }
  
        unsigned int numBlocksA = ((current_layer->output_size * current_layer->input_size) + blockSize - 1) / blockSize;
        if(grad_optimizer == "adam"){
          blap::gpu::updateWeightsDeltaAdam<<<numBlocksA,blockSize>>>(
              current_layer->d_weight_delta, current_layer->d_prev_weight_delta,
              current_layer->output_size, current_layer->input_size,
              input_vector, current_layer->d_delta,
              current_layer->bias_enabled, current_layer->bias,
              learning_rate, lambda,
              beta1, beta2, eta, epoch + 1,
              current_layer->d_moment_first, current_layer->d_moment_second
              );
        }
        else{
          blap::gpu::updateWeightsDeltaSgd<<<numBlocksA,blockSize>>>(
              current_layer->d_weight_delta, current_layer->d_prev_weight_delta,
              current_layer->output_size, current_layer->input_size,
              input_vector, current_layer->d_delta,
              current_layer->bias_enabled, current_layer->bias,
              learning_rate, lambda
              );
        }
        cudaDeviceSynchronize();
        checkGpuStatus();
        if(blap::trm){ break; }
  
        if(debug){
          cudaMemcpy(current_layer->weight_delta, current_layer->d_weight_delta, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyDeviceToHost);
          cout << "weight_delta : ";
          printMatrix(current_layer->output_size, current_layer->input_size, current_layer->weight_delta);
        }

        //XXX Special check, comment out
        //beta::gpu::checkDelta<<<1,1>>>(
        //    current_layer->d_weight_delta,
        //    current_layer->output_size,
        //    current_layer->input_size);
  
        // Updated Weights
        unsigned int numBlocksB = ((current_layer->output_size * current_layer->input_size) + blockSize - 1) / blockSize;
        blap::gpu::updateMatrix<<<numBlocksB,blockSize>>>(
            current_layer->d_weight,
            current_layer->d_weight_delta,
            current_layer->d_prev_weight_delta,
            current_layer->output_size,
            current_layer->input_size,
            current_layer->d_momentum,
            momentum_missmatch,
            momentum_start,
            momentum_max,
            momentum_step,
            momentum_rho,
            learning_rate,
            weight_decay);
        cudaDeviceSynchronize();
        checkGpuStatus();
        if(blap::trm){ break; }
  
        if(debug){
          cudaMemcpy(current_layer->weight, current_layer->d_weight, current_layer->output_size * current_layer->input_size * sizeof(double), cudaMemcpyDeviceToHost);
          cout << "updated weight : ";
          printMatrix(current_layer->output_size, current_layer->input_size, current_layer->weight);
        }
  
        // Update error for next layer up
        if(current_layer->prev_layer == NULL){ break; }
        unsigned int numErrorBlocks = ((current_layer->input_size) + blockSize - 1) / blockSize;
        blap::gpu::updateErrorUp<<<numErrorBlocks,blockSize>>>(
            current_layer->prev_layer->d_error,
            current_layer->d_error,
            current_layer->d_weight,
            current_layer->input_size,
            current_layer->output_size);
        cudaDeviceSynchronize();
        checkGpuStatus();
        
        // Move up the chain
        current_layer = current_layer->prev_layer;
      }

    } // Loop through sample_index
    if(blap::trm){ break; }
  }
  if(verbose and !debug){ cerr << "\n"; }
  
  cudaFree(d_sample_vector);
  free(sample_vector);

  cudaFree(d_sample_output);
  free(sample_output);
}

void Model::error(double &regression_error, double &classification_error, vector<DataItem> &test_set){
  double sum_errors = 0;
  int error_count = 0;
  for(unsigned int i = 0; i < test_set.size(); i++){
    if(blap::trm){ return; }

    double *result = this->solve(test_set.at(i).input);

    for(int j = 0; j < this->back_layer->output_size; j++){
      double *result = this->solve(test_set.at(i).input);
      sum_errors += ( (result[j] - test_set.at(i).output[j]) * (result[j] - test_set.at(i).output[j]) );
    }

    for(int j = 0; j < this->back_layer->output_size; j++){
      if(result[j] < 0.5 and test_set.at(i).output[j] < 0.5){ continue; }
      else if(result[j] > 0.5 and test_set.at(i).output[j] > 0.5){ continue; }
      else{ error_count++; break; }
    }
  }
  regression_error = sum_errors / ((double)test_set.size() * (double)this->back_layer->output_size);

  classification_error = (double)error_count / (double)test_set.size();

  return;
}

double Model::regression_error(vector<DataItem> &test_set){
  double sum_errors = 0;
  for(unsigned int i = 0; i < test_set.size(); i++){
    if(blap::trm){ return -999; }
    double *result = this->solve(test_set.at(i).input);
    for(int j = 0; j < this->back_layer->output_size; j++){
      sum_errors += ( (result[j] - test_set.at(i).output[j]) * (result[j] - test_set.at(i).output[j]) );
    }
  }
  double reg_error = sum_errors / ((double)test_set.size() * (double)this->back_layer->output_size);
  if(debug){
    cout << "regression error " << reg_error << " test_set.size " << test_set.size() << "\n";
  }

  return reg_error;
}

double Model::classification_error(vector<DataItem> &test_set){
  int error_count = 0;
  for(unsigned int i = 0; i < test_set.size(); i++){
    if(blap::trm){ return -999; }
    double *result = this->solve(test_set.at(i).input);
    
    if(debug){
      cout << "\n";
      printVector(this->back_layer->output_size,result);
      printVector(this->back_layer->output_size,test_set.at(i).output);
    }
    
    for(int j = 0; j < this->back_layer->output_size; j++){
      if(result[j] < 0.5 and test_set.at(i).output[j] < 0.5){ continue; }
      else if(result[j] > 0.5 and test_set.at(i).output[j] > 0.5){ continue; }
      else{ error_count++; break; }
    }
  }
  double error_percent = (double)error_count / (double)test_set.size();
  if(debug){
    cout << "error_count "<<error_count<<
      " test_set.size "<< test_set.size()<<
      " error "<< error_percent << "\n";
  }
  return error_percent;
}

void Model::writeLogFile(
    double test_reg_error, double test_class_error,
    double train_reg_error, double train_class_error,
    unsigned int test_size, unsigned int train_size){
  ofstream outStream;
  outStream.exceptions ( ifstream::failbit | ifstream::badbit );
  // Check if file exists, die otherwise
  try {
    outStream.open(this->log_file, std::ios::out | std::ios::app );
  }
  catch (const ifstream::failure& e) {
    cerr << " \033[31mERROR:\033[0m opening/writing file '" << 
      this->log_file << 
      "' : " << strerror(errno) <<
      "\n";
    exit(1);
  }
  outStream.exceptions ( ifstream::badbit );

  time_t endTime = time(nullptr);
  time_t runTime = endTime - startTime;

  // epoch,cumulative_runtime,test_regression_error,test_classification_error,train_regression_error,train_classification_error,test_size,train_size
  outStream << epoch << "," << runTime
    << "," << test_reg_error << "," << test_class_error 
    << "," << train_reg_error << "," << train_class_error 
    << "," << test_size << "," << train_size << "\n";
  outStream.close();
}

void Model::writeLogFile(double test_reg_error, double train_reg_error, unsigned int test_size, unsigned int train_size){
  ofstream outStream;
  outStream.exceptions ( ifstream::failbit | ifstream::badbit );
  // Check if file exists, die otherwise
  try {
    outStream.open(this->log_file, std::ios::out | std::ios::app );
  }
  catch (const ifstream::failure& e) {
    cerr << " \033[31mERROR:\033[0m opening/writing file '" << 
      this->log_file << 
      "' : " << strerror(errno) <<
      "\n";
    exit(1);
  }
  outStream.exceptions ( ifstream::badbit );

  time_t endTime = time(nullptr);
  time_t runTime = endTime - startTime;

  // epoch,cumulative_runtime,test_regression_error,test_classification_error,train_regression_error,train_classification_error,test_size,train_size
  outStream << epoch << "," << runTime
    << "," << test_reg_error 
    << ",," << train_reg_error 
    << ",," << test_size << "," << train_size << "\n";
  outStream.close();
}

void Model::setLogFile(string set_log_file){
  this->log_file_set = true;
  this->log_file = set_log_file;

  ofstream outStream;
  outStream.exceptions ( ifstream::failbit | ifstream::badbit );
  // Check if file exists, die otherwise
  try {
    outStream.open(this->log_file);
  }
  catch (const ifstream::failure& e) {
    cerr << " \033[31mERROR:\033[0m opening/writing file '" << 
      set_log_file << 
      "' : " << strerror(errno) <<
      "\n";
    exit(1);
  }
  outStream.exceptions ( ifstream::badbit );
  outStream << "epoch,cumulative_runtime,test_regression_error,test_classification_error,train_regression_error,train_classification_error,test_size,train_size\n";
  outStream.close();
}

