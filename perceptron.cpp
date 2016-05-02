
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>

using namespace::std;

double phi(double x) { return tanh(x);}
double Dphi(double x) { return (4*cosh(x)*cosh(x))/((cosh(2*x)+1)*(cosh(2*x)+1));}
double dot(const vector<double>& v1, const vector<double>& v2)
{
	double d=0.0;
	for(int i=0;i<v1.size();++i)
		d+= v1[i]*v2[i];
	return d;
}

vector<double> error(double u,vector<double> x, const vector<double>& w)
{
	double wx = dot(w,x);
	double e =  -2*(u - phi(wx))*Dphi(wx);
	for(int i=0;i<x.size();++i) x[i] *= e;
	return x;
}

int main()
{
	int Ntest = 10000;
	int Nit = 10000000;
	int Nin = 2;
	double eta = 0.001;
	vector<double> w(Nin+1);
	for(int i=0;i<Nin;++i) w[i] = rand()/double(RAND_MAX);


	vector<double> x(Nin);
	vector<double> dw(Nin);
	double u;
	double xsum;
	for(int it=0;it<Nit;++it) {
		xsum = 0.0;
		for(int i=0;i<Nin;++i){
			x[i] = rand()/double(RAND_MAX);
			xsum += x[i];
		}
		if(xsum>0.) u = 1;
		else u = -1;

		dw = error(u,x,w);
		for(int i=0;i<Nin;++i)
			w[i] -= eta*dw[i];
	
	}	
	int E =0;
	for(int it=0;it<Ntest;++it) {
		xsum = 0.0;
		for(int i=0;i<Nin;++i){
			x[i] = rand()/double(RAND_MAX);
			xsum += x[i];
		}
		if(xsum>0) u = 1;
		else u = -1;

		double out;
		if(phi(dot(x,w)) > .5) out=1;
		else out = -1;

//		cout << "target= " << u << endl;
//		cout << "value=  " << out << endl << endl;
		if(out != u) E += 1;	
	}	

	cout << double(E)/double(Ntest) << endl;

	return 0;
}

