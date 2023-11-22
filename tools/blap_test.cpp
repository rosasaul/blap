//  blap - deep learning library, training command line utility
//  Copyright (C) 2017  Saul Rosa <saul@megaframe.org>
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

#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <getopt.h>
#include <thread> // threads
#include <mutex> // mutex locking
#include <vector> // adjustable array
#include <time.h> // time interface
#include "blap/Model.h"
#include "blap/Utils.h"

using namespace std;

// Options and static variables
static int help_menu = 0;
static int verbose = 1;
static int debug = 0;

// Global use variables

// Start of main
int main(int argc, char** argv) {
  string test_file;
  string model_file;
  string output_file;
  int c;
  while(1){
    static struct option long_options[] = {
      {"help",       no_argument, &help_menu, 1},
      {"verbose",    no_argument, &verbose, 1},
      {"no-verbose", no_argument, &verbose, 0},
      {"test",       required_argument, 0, 't'},
      {"model",      required_argument, 0, 'm'},
      {"output",        required_argument, 0, 'o'},
      {"debug",      no_argument, &debug,   1},
      {0, 0, 0, 0}
    };
    int option_index = 0;
    c = getopt_long(argc, argv, "hvt:m:o:d",long_options, &option_index);
    if (c == -1){ break; }
    else if (c == 'h'){ help_menu = 1; }
    else if (c == 'v'){ verbose = 1; }
    else if (c == 't'){ test_file = optarg; }
    else if (c == 'm'){ model_file = optarg; }
    else if (c == 'o'){ output_file = optarg; }
    else if (c == 'd'){ debug = 1; }
    else if (c == '?'){ help_menu = 1; cout << "\n"; }
  }
  if(help_menu == 1){
    cout << "\nblap v" << BLAP_MAJOR_VERSION << "."<< BLAP_MIN_VERSION 
      << " deep learning library, test command line utility\n\n"
      << "REQUIRED:\n"
      << " -m, --model      model file, includes model structure, training configs, must include Weight Matrices\n"
      << " -t, --test       test input file\n"
      << "\n"
      << "OPTIONS:\n"
      << " -h, --help      Print this help menu\n"
      << " -v, --verbose   No screen prints, default on\n"
      << "  --no-verbose\n"
      << " -o, --output     Output results from model (same order as test input)\n"
      << " -d, --debug     Print additional debug statements\n\n"
    ;
    return 0;
  }

  // Check required, exit it missing items
  bool errors = false;
  if(test_file.empty()){
    cerr << " \033[31mERROR:\033[0m No test data file specified. See help menu.\n";
    errors = true;
  }
  if(model_file.empty()){
    cerr << " \033[31mERROR:\033[0m No model file specified. See help menu.\n";
    errors = true;
  }
  if(errors){ return 1; }

  // Log the start time
  time_t startTime = time(nullptr);

  // Create the blap interface
  blap::Model model;
  if(verbose){ model.verbose = true; }
  else{ model.verbose = false; }

  cerr << "Setup Model.\n";

  // Setup the Model settings from a file
  model.setup_model(model_file);

  // Check that the model had weight matrixes
  if(!model.pre_loaded_model){
    cerr << " \033[31mERROR:\033[0m Model file did not include weight matrices.\n";
    exit(1);
  }

  if(blap::debug){
    cerr << "printModel\n";
    model.printModel();
  }

  // Read in data sets
  vector<blap::DataItem> test_set = model.read_data(test_file);

  // Set the Cntrl-C handler (in utils.h)
  signal (SIGINT, blap::sigTerm);

  // Check test set
  cerr << "Check Test Set:\n";
  double reg_error = model.regression_error(test_set);
  double classification_error = model.classification_error(test_set);
  fprintf(stderr,"Classification error = %.2f%% Regression error = %.2f%%\n",classification_error * 100, reg_error * 100);
  if(!output_file.empty()){
    ofstream outStream;
    outStream.open(output_file);
    int output_size = model.back_layer->output_size;
    for(unsigned int i = 0; i < test_set.size(); i++){
      double *result = model.solve(test_set.at(i).input);
      outStream << result[0];
      for(int j = 1; j < output_size; j++){
        outStream<<"\t"<<result[j];
      }
      for(int j = 0; j < output_size; j++){
        outStream<<"\t"<<test_set.at(i).output[j];
      }
      outStream<<"\n";
    }
    outStream.close();
  }

  model.delete_data(test_set);

  time_t endTime = time(nullptr);
  time_t runTime = endTime - startTime;

  cerr << "blap exiting, run time " + to_string(runTime) + " seconds\n";

  return 0;
}

