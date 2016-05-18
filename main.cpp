#include "net.h"

#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <math.h>

using namespace::std;

typedef vector<int> veci;


int main()
{
	double eta = 0.1;
	int Nin = 12;
	int Nout = 1;
	int Nlayer = 8;
	int layerSize = 5;
	Net net(Nin,Nout,Nlayer,layerSize);

	vector<vector<double> > input(4,vector<double>(Nin,0.0));
	input[0][0] = -1.;
	input[0][1] = 1;
	input[1][0] = 1;
	input[1][1] = -1;
	input[2][0] = -1;
	input[2][1] = -1;
	input[3][0] = 1;
	input[3][1] = 1;

	vector<double> out(Nout,0.0);

	vector<vector<double> > target(4,vector<double>(Nout,0.0));
	target[0][0] = 1;
	target[1][0] = 1;
	target[2][0] = -1;
	target[3][0] = -1;

	net.train(input,target,eta,2000,true);

	net.result(input[1],out);
	for(int i=0;i<out.size();++i) 
		cout << out[i]<<  endl;



	return 0;
}
