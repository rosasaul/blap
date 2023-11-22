/* (c) Saul Rosa
 * Created 11.03.2015
 * Blap Model Class
 */

#ifndef blap_Utils_h_
#define blap_Utils_h_

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <utility>
#include <sys/stat.h>
#include <math.h>
#include <signal.h>

using namespace std;

namespace blap {

  extern bool trm;
  extern bool debug;
  void sigTerm(int param);
  extern unsigned int sigTerms;

  void printVector(unsigned int rows, double *vec);
  void printMatrix(unsigned int rows, unsigned int columns, double **matrix);
  void printMatrix(unsigned int rows, unsigned int columns, double *matrix);

  string formatted(time_t time);
}

#endif  // blap_Utils_h_


