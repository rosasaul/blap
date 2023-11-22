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
#include "blap/Model_gpu.h"
#include "blap/Utils.h"

using namespace std;

// Options and static variables
static int help_menu = 0;
static int verbose = 1;
static int debug = 0;
static int no_check = 0;

// Global use variables

// Start of main
int main(int argc, char** argv) {
  string train_file;
  string test_file;
  string validate_file;
  string model_file;
  string save_file;
  string csv_file;
  string log_file;
  unsigned int epoch_interval = 0;
  unsigned int gpuId = 0;
  int c;
  while(1){
    static struct option long_options[] = {
      {"help",       no_argument, &help_menu, 1},
      {"verbose",    no_argument, &verbose, 1},
      {"no-verbose", no_argument, &verbose, 0},
      {"train",      required_argument, 0, 'r'},
      {"test",       required_argument, 0, 't'},
      {"validate",   required_argument, 0, 'a'},
      {"model",      required_argument, 0, 'm'},
      {"save",       required_argument, 0, 's'},
      {"csv",        required_argument, 0, 'o'},
      {"log",        required_argument, 0, 'l'},
      {"gpu",        required_argument, 0, 'g'},
      {"epoch-int",  required_argument, 0, 'e'},
      {"no-check",   no_argument, &no_check, 1},
      {"debug",      no_argument, &debug,   1},
      {0, 0, 0, 0}
    };
    int option_index = 0;
    c = getopt_long(argc, argv, "hvr:t:a:m:s:o:l:g:e:nd",long_options, &option_index);
    if (c == -1){ break; }
    else if (c == 0){ continue; }
    else if (c == 'h'){ help_menu = 1; }
    else if (c == 'v'){ verbose = 1; }
    else if (c == 'r'){ train_file = optarg; }
    else if (c == 't'){ test_file = optarg; }
    else if (c == 'a'){ validate_file = optarg; }
    else if (c == 'm'){ model_file = optarg; }
    else if (c == 's'){ save_file = optarg; }
    else if (c == 'o'){ csv_file = optarg; }
    else if (c == 'l'){ log_file = optarg; }
    else if (c == 'g'){ gpuId = stoi(optarg); }
    else if (c == 'e'){ epoch_interval = stoi(optarg); }
    else if (c == 'n'){ no_check = 1; }
    else if (c == 'd'){ debug = 1; }
    else if (c == '?'){ help_menu = 1; cout << "\n"; }
    else{
      cerr << "\033[31mERROR:\033[0m Uknown option '" << c << "'.\n";
      exit(1);
    }
  }
  if(help_menu == 1){
    cout << "\nblap v" << BLAP_MAJOR_VERSION << "."<< BLAP_MIN_VERSION 
      << " deep learning library, training command line utility\n\n"
      << "REQUIRED:\n"
      << " -r, --train      train input file\n"
      << " -t, --test       test input file\n"
      << " -a, --validate   validation input file\n"
      << " -m, --model      model file, includes model structure, training configs\n"
      << " -s, --save       file name to save the model to when done training\n"
      << "\n"
      << "OPTIONS:\n"
      << " -h, --help       Print this help menu\n"
      << " -l, --log        Log File with Classification and Regression error\n"
      << " -g, --gpu        Specify gpu to use, default 0\n"
      << " -v, --verbose    No screen prints, default on\n"
      << "  --no-verbose\n"
      << " -e --epoch-int   Epoch interval, default 10\n"
      << " -n --no-check    No Train/Test/Validation checks\n"
      << " -d, --debug      Print additional debug statements\n\n"
    ;
    return 0;
  }

  // Check required, exit it missing items
  bool errors = false;
  if(train_file.empty()){
    cerr << " \033[31mERROR:\033[0m No train data file specified. See help menu.\n";
    errors = true;
  }
  if(test_file.empty()){
    cerr << " \033[31mERROR:\033[0m No test data file specified. See help menu.\n";
    errors = true;
  }
  if(validate_file.empty()){
    cerr << " \033[31mERROR:\033[0m No validate data file specified. See help menu.\n";
    errors = true;
  }
  if(model_file.empty()){
    cerr << " \033[31mERROR:\033[0m No model file specified. See help menu.\n";
    errors = true;
  }
  if(errors){ return 1; }

  if(debug == 1){ blap::debug = true; }

  // Log the start time
  time_t startTime = time(nullptr);
  srand(startTime); // seed random

  // Create the blap interface
  blap::Model model(gpuId);

  if(verbose){ model.verbose = true; }
  else{ model.verbose = false; }

  if(no_check == 1){ model.no_check = true; }

  if(!log_file.empty()){ model.setLogFile(log_file); }

  fprintf(stderr, "[%s] Setup Model...\n", blap::formatted(time(nullptr)).c_str());

  // Setup the Model settings from a file
  model.setup_model(model_file);

  // Set interval of epoch prints
  if(epoch_interval > 0){ model.epoch_interval = epoch_interval; }

  if(blap::debug){
    cerr << "Print Model:\n";
    model.printModel();
  }

  // Read in data sets
  fprintf(stderr, "[%s] Read in train dataset...\n", blap::formatted(time(nullptr)).c_str());
  vector<blap::DataItem> training_set = model.read_data(train_file);

  fprintf(stderr, "[%s] Read in test dataset...\n", blap::formatted(time(nullptr)).c_str());
  vector<blap::DataItem> test_set = model.read_data(test_file);

  fprintf(stderr, "[%s] Read in validate dataset...\n", blap::formatted(time(nullptr)).c_str());
  vector<blap::DataItem> validate_set = model.read_data(validate_file);

  // Set the Ctrl-C handler (in utils.h)
  signal(SIGINT, blap::sigTerm);

  // Initialize Training
  fprintf(stderr, "[%s] Training...\n", blap::formatted(time(nullptr)).c_str());
  model.train(training_set,test_set);

  if(blap::debug){
    cerr << "Print Model:\n";
    model.printModel();
  }

  if(!blap::trm and no_check == 0){
    cerr << "Check Validation Set: ";
      double classification_error = model.classification_error(validate_set);
    double reg_error = model.regression_error(validate_set);
    fprintf(stderr,"Classification error = %.2f%% Regression error = %.2f%%\n",classification_error * 100, reg_error * 100);
  }

  time_t endTime = time(nullptr);
  time_t runTime = endTime - startTime;

  if(!save_file.empty()){ model.save(save_file); }

  model.delete_data(training_set);
  model.delete_data(test_set);
  model.delete_data(validate_set);

  cerr << "blap exiting, run time " + to_string(runTime) + " seconds\n";

  return 0;
}

