/* (c) Saul Rosa
 * Created 11.03.2015
 * Blap Model Class
 */

#ifndef blap_ActivationFunctions_h_
#define blap_ActivationFunctions_h_

#include <math.h>

using namespace std;

namespace blap {

  double sigmoid_solver(double x);
  double sigmoid_derivative(double x);

  double tanh_solver(double x);
  double tanh_derivative(double x);

  double ELU_solver(double x);
  double ELU_derivative(double x);

  double ReLU_solver(double x);
  double ReLU_derivative(double x);

  double leakyReLU_solver(double x);
  double leakyReLU_derivative(double x);

  double linear_solver(double x);
  double linear_derivative(double x);

}

#endif  // blap_ActivationFunctions_h_


