/* (c) Saul Rosa
 * Created 11.03.2015
 * Blap Model Class
 */

#ifndef blap_Model_h_
#define blap_Model_h_

#define BLAP_MAJOR_VERSION 0
#define BLAP_MIN_VERSION 1

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <utility>
#include <sys/stat.h>
#include <math.h>
#include "blap/ActivationFunctions.h"
#include "blap/InitFunctions.h"
#include "blap/Utils.h"

using namespace std;

namespace blap {

  struct Layer {
    int input_size;
    int output_size;
    string activiation_function;
    double bias;
    bool bias_enabled = false;
    double** matrix;
    double** weight_delta;
    double** momentum;
    Layer *next_layer = NULL;
    Layer *prev_layer = NULL;
    double *output_activation;
    double *output_activation_derivative;
  };

  struct DataItem {
    double * input;
    double * output;
  };

  typedef double (*ActivationPointer)(double);
  typedef double (*InitPointer)(double);

  class Model {
    private:
      double *mulVecMat(double *input, double **matrix, int row, int col);
    public:
      double max_weight = 0.5;
      string weight_init_func = "rand";
      double learning_rate = 0.1;
      double momentum_matching = 0.05;
      double momentum_missmatch = 0.5;
      double momentum_max = 2.5;
      double stop_threshold = 0.0001;
      double max_iterations = 3000;
      double momentum_step = 0.1;
      double momentum_start = 0.05;
      bool pre_loaded_model = false;
      bool verbose = false;

      map<string, ActivationPointer> activation_function_map;
      map<string, InitPointer> weight_init_function_map;
      int input_size;
      Layer *front_layer;
      Layer *back_layer;

      Model();
      ~Model();

      vector<DataItem> read_data(string data_file);
      void delete_data(vector<blap::DataItem> test_set);
      
      void setup_model(string model_file);
      void train(vector<DataItem> training_set, vector<DataItem> test_set);
      double *solve(double *input);
      double classification_error(vector<DataItem> test_set);
      double regression_error(vector<DataItem> test_set);
      void printModel();
      void save(string save_file);
  };

}

#endif  // blap_Model_h_


