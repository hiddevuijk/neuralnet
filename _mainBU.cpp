#include "netBU.h"

#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <math.h>

using namespace::std;

typedef vector<int> veci;


int main()
{
	double eta = .1;
	int L = 50;
	int N = 50;
	vector<int> a(L);
	a[0] = 2;
	for(int i=1;i<L;++i) a[i] = N;
	a.back() = 1;
	Net net(a);

	vector<vector<double> > input(4,vector<double>(2,0.0));
	input[0][0] = -1.;
	input[0][1] = 1;
	input[1][0] = 1;
	input[1][1] = -1;
	input[2][0] = -1;
	input[2][1] = -1;
	input[3][0] = 1;
	input[3][1] = 1;

	vector<double> out(a.back());

	vector<vector<double> > target(4,vector<double>(1,0.00));
	target[0][0] = 1;
	target[1][0] = 1;
	target[2][0] = -1;
	target[3][0] = 0;

	vector<double> errors;	

	for(int i=0;i<200;++i) {
		int n = rand()%3;
		net.feedForward(input[n],out);
		errors.push_back(error(out,target[n]));
		net.feedBack(target[n]);
		net.newW(eta);

	}

	net.feedForward(input[2],out);
	for(int i=0;i<out.size();++i) 
		cout << out[i]<<  endl;

	return 0;
}
