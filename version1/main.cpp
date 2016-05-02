#include "neuralnet.h"

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <math.h>

using namespace::std;

double f(double x) { return tanh(x);}
double df(double x) { return 1-tanh(x)*tanh(x);}

int main()
{
	int N = 100;

	Net net(2,2,3,5,f,df);
	vector<double> in(2,0.0);
	vector<double> out(2,0.0);
	vector<double> target(2,0.0);
	in[0] = 1.;
	in[1] = -1.;
	target[0] = 1.;
	target[1] = -1.;

	net.feedForward(in,out);

	for(int i=0;i<out.size();++i)
		cout << out[i] << '\t';
	cout << endl;
	for(int i=0;i<N;++i) {
		net.train(in,target,0.1);
	}
	net.feedForward(in,out);

	
	for(int i=0;i<out.size();++i)
		cout << out[i] << '\t';
	cout << endl;

	return 0;
}


