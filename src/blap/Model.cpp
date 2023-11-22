//  blap - deep learning library
//    "blap, sometimes it just hits you in the face."
//
//  Copyright (C) 2017  Saul Rosa http://www.megaframe.org
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

#include "blap/Model.h"
#include "blap/Utils.h"

using namespace std;
using namespace blap;

Model::Model(){
  activation_function_map["sigmoid"] = sigmoid_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["sigmoid_derivative"] = sigmoid_derivative;
  
  activation_function_map["tanh"] = tanh_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["tanh_derivative"] = tanh_derivative;
 
  activation_function_map["ELU"] = ELU_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["ELU_derivative"] = ELU_derivative;
  
  activation_function_map["ReLU"] = ReLU_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["ReLU_derivative"] = ReLU_derivative;
  
  activation_function_map["leakyReLU"] = ReLU_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["leakyReLU_derivative"] = ReLU_derivative; 
  
  activation_function_map["linear"] = linear_solver; // Usage : double y = map["sigmoid"](0.1);
  activation_function_map["linear_derivative"] = linear_derivative; 


  weight_init_function_map["rand"] = rand_init;
  weight_init_function_map["ones"] = ones_init;
}

Model::~Model(){
  // Delete insantiated layers
  Layer *current_layer = front_layer;
  Layer *prev_layer;
  while(current_layer != NULL){

    for(int i = 0; i < current_layer->output_size; i++){
      delete[] current_layer->matrix[i];
      delete[] current_layer->weight_delta[i];
      delete[] current_layer->momentum[i];
    }
    if(current_layer->output_activation){
      delete[] current_layer->output_activation;
    }
    if(current_layer->output_activation_derivative){
      delete[] current_layer->output_activation_derivative;
    }
    delete[] current_layer->matrix;
    delete[] current_layer->weight_delta;
    delete[] current_layer->momentum;

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
  outStream<<"learning_rate=" << learning_rate << "\n";
  outStream<<"momentum_matching=" << momentum_matching << "\n";
  outStream<<"momentum_missmatch=" << momentum_missmatch << "\n";
  outStream<<"momentum_max=" << momentum_max << "\n";
  outStream<<"max_weight=" << max_weight << "\n";
  outStream<<"stop_threshold=" << stop_threshold << "\n";
  outStream<<"max_iterations=" << max_iterations << "\n";
  outStream<<"momentum_step=" << momentum_step << "\n";
  outStream<<"momentum_start=" << momentum_start << "\n";
  outStream<<"weight_init_func="<< weight_init_func <<"\n";
  outStream<<"input_size=" << input_size << "\n";

  Layer *current_layer = front_layer;
  while(current_layer != NULL){
    outStream<<"layer:\n";
    outStream<<"  output_size="<< current_layer->output_size <<"\n";
    outStream<<"  activiation_function=" << current_layer->activiation_function << "\n";
    if(current_layer->bias_enabled){ outStream << "  bias=" << current_layer->bias << "\n"; }

    for(int i = 0; i < current_layer->output_size; i++){
      outStream<<"  [";
      for(int j = 0; j < current_layer->input_size; j++){
        outStream << "\t" << current_layer->matrix[i][j];
      }
      outStream<<"]\n";
    }

    current_layer = current_layer->next_layer;
  }
  outStream.close();
}

void Model::setup_model(string model_file){
  ifstream inStream;
  inStream.exceptions ( ifstream::failbit | ifstream::badbit );

  smatch match_results;
  regex comment("#.*");
  regex re_learning_rate("^learning_rate=(\\S+)");
  regex re_momentum_matching("^momentum_matching=(\\S+)");
  regex re_momentum_missmatch("^momentum_missmatch=(\\S+)");
  regex re_momentum_max("^momentum_max=(\\S+)");
  regex re_max_weight("^max_weight=(\\S+)");
  regex re_stop_threshold("^stop_threshold=(\\S+)");
  regex re_max_iterations("^max_iterations=(\\S+)");
  regex re_momentum_step("^momentum_step=(\\S+)");
  regex re_momentum_start("^momentum_start=(\\S+)");
  regex re_weight_init_func("^weight_init_func=(\\S+)");
  regex re_input_size("^input_size=(\\S+)");
  regex re_layer_start("^layer:");
  regex re_output_size("^  output_size=(\\S+)");
  regex re_activiation_function("^  activiation_function=(\\S+)");
  regex re_bias("^  bias=(\\S+)");
  regex re_matrix_line("^  \\["); // \\.-\t
  regex re_matrix_data("\\[\t(.*?)\\]");
  regex re_non_layer("^\\S+");

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

  int layer_count = 0;
  int current_input_size = 0;
  
  // Read in Model structure
  string line;
  bool redo_line = false;
  while( redo_line or getline(inStream,line) ){
    redo_line = false;
    line = regex_replace(line,comment,"");
    if(regex_search(line,match_results,re_learning_rate)){
      learning_rate = stod(match_results[1].str().c_str());
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
    else if(regex_search(line, match_results, re_weight_init_func)){
      weight_init_func = match_results[1].str();
    }
    else if(regex_search(line, match_results, re_input_size)){
      input_size = stod(match_results[1].str().c_str());
      current_input_size = input_size;
    }
    else if(regex_search(line, match_results, re_layer_start)){
      int i = 0;
      while( getline(inStream,line) ){
        if(regex_search(line, match_results, re_output_size)){
          current_layer->output_size = stoi(match_results[1].str().c_str());
          current_layer->matrix = new double*[current_layer->output_size];
          current_layer->weight_delta = new double*[current_layer->output_size];
          current_layer->momentum = new double*[current_layer->output_size];
        }
        else if(regex_search(line, match_results, re_activiation_function)){
          current_layer->activiation_function = match_results[1].str();
        }
        else if(regex_search(line, match_results, re_bias)){
          current_layer->bias = stod(match_results[1].str().c_str());
          current_layer->bias_enabled = true;
          current_input_size++;
        }
        else if(regex_search(line, match_results, re_matrix_line)){
          if(regex_search(line, match_results, re_matrix_data)){
            pre_loaded_model = true;
            string matrix_line = match_results[1].str();
            current_layer->matrix[i] = new double[current_input_size];
            current_layer->weight_delta[i] = new double[current_input_size];
            current_layer->momentum[i] = new double[current_input_size];

            string delimiter = "\t";
            int j = 0;
            size_t pos = 0;
            double token;
            while ((pos = matrix_line.find(delimiter)) != string::npos) {
              token = stod( matrix_line.substr(0, pos).c_str() );
              matrix_line.erase(0, pos + delimiter.length());
              current_layer->matrix[i][j] = token;
              j++;
              if(j >= current_input_size){
                cerr << " \033[31mERROR:\033[0m Layer Matrix count mismatch, "<<
                  "extra data found at layer "<<layer_count<<" row "<< i << " column "<<j<<"\n";
                exit(1);
              }
            }
            token = stod( matrix_line.c_str() );
            current_layer->matrix[i][j] = token;
            i++;
          }
          else{
            cerr << " \033[31mERROR:\033[0m Malformed Matrix.\n";
            exit(1);
          }
        }
        else if(
            regex_search(line, match_results, re_layer_start) or
            regex_search(line, match_results, re_non_layer)
            ){
          // Save off the input size
          current_layer->input_size = current_input_size;
          
          // Update weight if no vector lines were specified
          if(i == 0){
            for(i = 0; i < current_layer->output_size; i++){
              current_layer->matrix[i] = new double[current_layer->input_size];
              current_layer->weight_delta[i] = new double[current_layer->input_size];
              current_layer->momentum[i] = new double[current_layer->input_size];
              for(int j = 0; j < current_layer->input_size; j++){
                current_layer->matrix[i][j] = weight_init_function_map[weight_init_func](max_weight);
                current_layer->weight_delta[i][j] = 0;
                current_layer->momentum[i][j] = momentum_start;
              }
            }
          }

          if( regex_search(line, match_results, re_non_layer) ){
            // Prepare for next Layer
            current_input_size = current_layer->output_size;
            current_layer->next_layer = new Layer();
            current_layer->next_layer->prev_layer = current_layer;
            current_layer = current_layer->next_layer;
            layer_count++;
          }
          break;
        }
      }

      // Save off the input size
      current_layer->input_size = current_input_size;

      // Catch last layer in case model file ends on layer
      if(i == 0){
        for(i = 0; i < current_layer->output_size; i++){
          current_layer->matrix[i] = new double[current_layer->input_size];
          current_layer->weight_delta[i] = new double[current_layer->input_size];
          current_layer->momentum[i] = new double[current_layer->input_size];
          for(int j = 0; j < current_layer->input_size; j++){
            current_layer->matrix[i][j] = weight_init_function_map[weight_init_func](max_weight);
            current_layer->weight_delta[i][j] = 0;
            current_layer->momentum[i][j] = momentum_start;
          }
        }
      }
      
      redo_line = true;
      continue;
    }
  }

  inStream.close();

  back_layer = current_layer;
  
  // TODO check setup (input/output sizes are set correctly)
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
  int line_num = 1;
  while( getline(inStream,line) ){
    DataItem data;
    data.input = new double[input_size];
    data.output = new double[back_layer->output_size];
    
    int token_idx = 0;
    size_t pos = 0;
    double token;
    while ((pos = line.find(delimiter)) != string::npos){
      token = stod( line.substr(0, pos).c_str() );
      line.erase(0, pos + delimiter.length());
      if(token_idx < input_size){
        data.input[token_idx] = token;
      }
      else if (token_idx - input_size < back_layer->output_size){
        data.output[token_idx - input_size] = token;
      }
      else{
        cerr << " \033[31mERROR:\033[0m Data line does not match model size, too many components on line "<< 
          line_num << "\n";
        exit(1);
      }
      token_idx++;
    }
    token = stod( line.c_str() );
    data.output[token_idx - input_size] = token;

    if(token_idx < input_size + back_layer->output_size - 1){
       cerr << " \033[31mERROR:\033[0m Data line does not match model size, insufficient components on line "<<
         line_num << " expecting " << input_size + back_layer->output_size << " recieved " << token_idx << "\n";
       exit(1);
    }

    dataset.push_back(data);
    line_num++;
  }

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
    blap::printMatrix(current_layer->output_size, current_layer->input_size, current_layer->matrix);
    current_layer = current_layer->next_layer;
  }
}

double *Model::solve(double *input){
  bool firstLayer = true;
  Layer *current_layer = front_layer;
  while(current_layer != NULL){
    double *output = mulVecMat(input, current_layer->matrix, current_layer->output_size, current_layer->input_size);

    int output_act_size = current_layer->output_size;
    if(current_layer->next_layer != NULL and current_layer->next_layer->bias_enabled){
      output_act_size++;
    }

    // Compute activation / activation derivative
    double *output_act = new double[output_act_size];
    for(int i = 0; i < current_layer->output_size; i++){
      output_act[i] = activation_function_map[
        current_layer->activiation_function
      ](output[i]);
    }
    delete[] output;
    if(current_layer->next_layer != NULL and current_layer->next_layer->bias_enabled){
      output_act[output_act_size - 1] = activation_function_map[
        current_layer->activiation_function
      ](current_layer->next_layer->bias);
    }

    // Save off activation/act derivative
    if(current_layer->output_activation){ delete[] current_layer->output_activation; }
    current_layer->output_activation = output_act;

    // Exit if at end of chain
    if(current_layer->next_layer == NULL){ break; }

    // Update to next layer
    current_layer = current_layer->next_layer;

    // Pass output to input of next layer
    if(!firstLayer){ delete[] input; }
    else{ firstLayer = false; }
    input = new double[current_layer->input_size];
    for(int i = 0; i < output_act_size; i++){
      input[i] = output_act[i];
    }
    if(current_layer->bias_enabled){
      input[current_layer->input_size - 1] = current_layer->bias;
    }
  }
  delete[] input;

  return current_layer->output_activation;
}


double *Model::mulVecMat(double *input, double **matrix, int row, int col){
  double *output = new double[row];
  for(int i = 0; i < row; i++){
    output[i] = 0;
    for(int j = 0; j < col; j++){
      output[i] += input[j] * matrix[i][j];
    }
  }
  return output;
}



void Model::train(vector<DataItem> training_set, vector<DataItem> test_set){
  vector<int> sample_indexs;
  for(unsigned int i = 0; i < training_set.size(); i++){ sample_indexs.push_back(i); }
  //random_shuffle(sample_indexs.begin(), sample_indexs.end());
  unsigned int select_idx = 0;
  unsigned int num_samples = sample_indexs.size();

  for(int epoch = 1; epoch <= max_iterations; epoch++){
    if(blap::trm){ break; }
    
    unsigned int sample_index = sample_indexs[select_idx]; //rand() % training_set.size();
    select_idx++;
    if(select_idx >= num_samples){ select_idx = 0; }

    Layer *current_layer = this->front_layer;
    
    double *input = new double[current_layer->input_size];
    
    for(int i = 0; i < this->input_size; i++){
      input[i] = training_set.at(sample_index).input[i];
    }
    if(current_layer->bias_enabled){
      input[current_layer->input_size - 1] = current_layer->bias;
    }

    if(debug){
      cout << "Train vector : " << sample_index << " : ";
      printVector(current_layer->input_size, input);
    }

    while(!blap::trm and current_layer != NULL){
      double *output = mulVecMat(input, current_layer->matrix, current_layer->output_size, current_layer->input_size);
      delete[] input;

      if(debug){
        cout << "output: ";
        printVector(current_layer->output_size, output);
      }

      // Compute activation / activation derivative
      double *output_act_der = new double[current_layer->output_size];
      double *output_act = new double[current_layer->output_size];
      for(int i = 0; i < current_layer->output_size; i++){
        output_act_der[i] = this->activation_function_map[
          current_layer->activiation_function + "_derivative"
        ](output[i]);

        output_act[i] = this->activation_function_map[
          current_layer->activiation_function
        ](output[i]);
      }
      delete[] output;

      if(debug){
        cout << "output_act: ";
        printVector(current_layer->output_size, output_act);
        cout << "output_act_der: ";
        printVector(current_layer->output_size, output_act_der);
      }

      // Save off activation/act derivative
      if(current_layer->output_activation){ delete[] current_layer->output_activation; }
      current_layer->output_activation = output_act;
      if(current_layer->output_activation_derivative){ delete[] current_layer->output_activation_derivative; }
      current_layer->output_activation_derivative = output_act_der;
      
      // Exit if at end of chain
      if(current_layer->next_layer == NULL){ break; }

      // Update to next layer
      current_layer = current_layer->next_layer;

      // Pass output to input of next layer
      input = new double[current_layer->input_size];
      for(int i = 0; i < current_layer->prev_layer->output_size; i++){
        input[i] = output_act[i];
      }
      if(current_layer->bias_enabled){
        input[current_layer->input_size - 1] = current_layer->bias;
      }
    }

    if(debug){
      cout << "output_activation: ";
      printVector(current_layer->output_size, current_layer->output_activation);
    }
    
    // Back propegation //
    // Get error at end of chain
    double *error = new double[current_layer->output_size];
    for(int i = 0; i < current_layer->output_size; i++){
      error[i] = current_layer->output_activation[i] - training_set.at(sample_index).output[i];
    }

    while(!blap::trm and current_layer != NULL){

      // Solve for delta
      double *delta = new double[current_layer->output_size];
      for(int i = 0; i < current_layer->output_size; i++){
        delta[i] = error[i] * current_layer->output_activation_derivative[i];
      }

      if(debug){
        cout << "error: ";
        printVector(current_layer->output_size, error);
        cout << "Delta: ";
        printVector(current_layer->output_size, delta);
      }

      double *input_vector = new double[current_layer->input_size];

      if(current_layer->prev_layer == NULL){
        for(int i = 0; i < this->input_size; i++){
          input_vector[i] = training_set.at(sample_index).input[i];
        }
      }
      else {
        for(int i = 0; i < current_layer->prev_layer->output_size; i++){
          input_vector[i] = current_layer->prev_layer->output_activation[i];
        }
      }

      if(current_layer->bias_enabled){
        input_vector[current_layer->input_size - 1] = current_layer->bias;
      }

      double **weight_delta = new double*[current_layer->output_size];
      for(int i = 0; i < current_layer->output_size; i++){
        weight_delta[i] = new double[current_layer->input_size];
        for(int j = 0; j < current_layer->input_size; j++){
          weight_delta[i][j] = -learning_rate * input_vector[j] * delta[i];
        }
      }
      delete[] input_vector; delete[] delta;
      if(debug){
        cout << "weight_delta : ";
        printMatrix(current_layer->output_size, current_layer->input_size, weight_delta);
      }

      // Updated Weights
      for(int i = 0; i < current_layer->output_size; i++){
        for(int j = 0; j < current_layer->input_size; j++){
          double momentum_add = 0;
          if(
              ( weight_delta[i][j] > 0 and current_layer->weight_delta[i][j] > 0 ) or
              ( weight_delta[i][j] < 0 and current_layer->weight_delta[i][j] < 0 )
          ){
            momentum_add = current_layer->momentum[i][j] * current_layer->weight_delta[i][j];
            if(current_layer->momentum[i][j] < momentum_max){ current_layer->momentum[i][j] += momentum_step; }
          }
          else{
            current_layer->momentum[i][j] = momentum_start;
            momentum_add = momentum_missmatch * current_layer->weight_delta[i][j];
          }
          current_layer->matrix[i][j] += weight_delta[i][j] + momentum_add;
        }
      }

      // Delete weight matrix
      for(int i = 0; i < current_layer->output_size; i++){ delete[] weight_delta[i]; }
      delete[] weight_delta;

      if(debug){
        cout << "updated matrix : ";
        printMatrix(current_layer->output_size, current_layer->input_size, current_layer->matrix);
      }

      // Update error for next layer up
      double *error_going_up = new double[current_layer->input_size];
      for(int j = 0; j < current_layer->input_size; j++){ error_going_up[j] = 0; }

      for(int i = 0; i < current_layer->output_size; i++){
        for(int j = 0; j < current_layer->input_size; j++){
          error_going_up[j] += error[i] * current_layer->matrix[i][j];
        }
      }
      if(error){ delete[] error; }
      error = error_going_up;

      // Move up the chain
      current_layer = current_layer->prev_layer;
    }
    if(error){ delete[] error; }

    // Check error
    if(epoch % 10 == 0){
      double test_error = this->classification_error(test_set);
      if(verbose){
        if(debug){
          cerr << "epoch " << epoch << " test_error " << test_error << "\n";
        }
        else if(epoch % 100 == 0){
          double reg_error = this->regression_error(test_set);
          fprintf(
            stderr,
            "\nEpoch %8d : Test Error %5.2f%% : Regression Error %5.2f%%\n",
            epoch,
            test_error * 100,
            reg_error * 100
          );
        }
        else{
          fprintf(
            stderr,
            "\rEpoch %8d : Test Error %5.2f%%",
            epoch,
            test_error * 100
          );
        }
      }
      if(test_error < stop_threshold){ break; }
    }
  }
  if(verbose and !debug){ cerr << "\n"; }
}

double Model::regression_error(vector<DataItem> test_set){
  double sum_errors = 0;
  for(unsigned int i = 0; i < test_set.size(); i++){
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

double Model::classification_error(vector<DataItem> test_set){
  int error_count = 0;
  for(unsigned int i = 0; i < test_set.size(); i++){
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

