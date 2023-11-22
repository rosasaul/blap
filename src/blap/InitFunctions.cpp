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

#include <stdlib.h>
#include "blap/InitFunctions.h"

using namespace std;


double blap::rand_init(double min_weight, double max_weight){
  return (((max_weight - min_weight) * ((double) rand() / (RAND_MAX))) + min_weight);
}

