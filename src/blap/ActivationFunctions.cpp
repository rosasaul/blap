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

#include "blap/ActivationFunctions.h"
#include <stdio.h>

using namespace std;
using namespace blap;

double blap::tanh_solver(double x){
  return ((tanh(x) + 1) / 2);
}

double blap::tanh_derivative(double x){
  return ((1 - (tanh(x) * tanh(x)))/2);
}

double blap::sigmoid_solver(double x){
  return 1 / ( 1 + exp(- x ) );
}

double blap::sigmoid_derivative(double x){
  double yx = sigmoid_solver(x);
  return yx * ( 1 - yx);
}

double blap::ReLU_solver(double x){
  if(x < 0){ return 0; }
  return x;
}

double blap::ReLU_derivative(double x){
  if(x < 0){ return 0; }
  return 1;
}

double blap::ELU_solver(double x){
  if(x < 0){ return 0.2 * (exp(x) - 1); }
  return x;
}

double blap::ELU_derivative(double x){
  if(x < 0){ return 0.2 * exp(x); }
  return 1;
}


double blap::leakyReLU_solver(double x){
  if(x < 0){ return 0.2 * x; }
  return x;
}

double blap::leakyReLU_derivative(double x){
  if(x < 0){ return 0.2; }
  return 1;
}

double blap::linear_solver(double x){
  return x;
}

double blap::linear_derivative(double x){
  return 1;
}

