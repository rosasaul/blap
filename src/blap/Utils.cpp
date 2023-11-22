//  blap - deep learning gpu accelerated library
//  Copyright (C) 2017  Saul Rosa
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

#include "blap/Utils.h"


namespace blap {
  using namespace std;

  bool trm = false;
  bool debug = false;
  unsigned int sigTerms = 0;

  /* Catches Cntrl-C and sets the global flag trm
   */
  void sigTerm(int param){
    if(sigTerms > 0){ exit(0); }
    trm = true;
    sigTerms++;
  }

  void printVector(unsigned int rows, double *vec){
    string output = "vector : [";
    for(unsigned int i = 0; i < rows; i++){
      output += " " + to_string(vec[i]);
    }
    output += " ]\n";
    cout << output;
  }
  
  void printMatrix(unsigned int rows, unsigned int columns, double **matrix){
    string output = "matrix : \n";
    for(unsigned int i = 0; i < rows; i++){
      if(rows == 1){ output += "["; }
      else if(i == 0 ){ output += "/"; }
      else if(i == rows - 1){ output += "\\"; }
      else{ output += "|"; }
      for(unsigned int j = 0; j < columns; j++){
        output += " " + to_string(matrix[i][j]);
      }
      if(rows == 1){ output += " ]\n"; }
      else if(i == 0 ){ output += " \\\n"; }
      else if(i == rows - 1){ output += " /\n"; }
      else{ output += " |\n"; }
    }
    cout << output;
  }

  void printMatrix(unsigned int rows, unsigned int columns, double *matrix){
    string output = "matrix : \n";
    for(unsigned int i = 0; i < rows; i++){
      if(rows == 1){ output += "["; }
      else if(i == 0 ){ output += "/"; }
      else if(i == rows - 1){ output += "\\"; }
      else{ output += "|"; }
      for(unsigned int j = 0; j < columns; j++){
        output += " " + to_string(matrix[i * columns + j]);
      }
      if(rows == 1){ output += " ]\n"; }
      else if(i == 0 ){ output += " \\\n"; }
      else if(i == rows - 1){ output += " /\n"; }
      else{ output += " |\n"; }
    }
    cout << output;
  }

  /* Format Time into mysql format
   * @param time unix timestam integer
   */
  string formatted(time_t time){
    struct tm * timeinfo;
    timeinfo = localtime(&time);

    char buffer [20];
    sprintf(buffer, "%d-%02d-%02d %02d:%02d:%02d",
      1900 + timeinfo->tm_year,
      timeinfo->tm_mon + 1,
      timeinfo->tm_mday,
      timeinfo->tm_hour,
      timeinfo->tm_min,
      timeinfo->tm_sec
    );  
    return string(buffer);
  }

}


