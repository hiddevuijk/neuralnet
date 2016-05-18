#include "aux.h"

#include <iostream>
#include <math.h>
#include <random>

using namespace::std;

int main()
{
	int N = 100;
	int n = 10;

	default_random_engine generator;
	normal_distribution<double> Ndist(0.0,1.0);


	vector<vector<double> > input(N,vector<double>(n));

	for(int Ni =0;Ni<N;++Ni) {
		for(int ni=0;ni<n;++ni) {
			input[Ni][ni] = Ndist(generator);
		}
	}
		
	rescale(input);





	return 0;
}




