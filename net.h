#ifndef GUARD_net_h
#define GUARD_net_h

#include <math.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

double phi(double input) { return tanh(input);}
double dphi(double input) { return 1-tanh(input)*tanh(input);}

double error(const std::vector<double>& output, const std::vector<double>& target)
{
	// assert 
	double e=0;
	for(int n=0;n<output.size();++n)
		e += (output[n]-target[n])*(output[n]-target[n]);
	return e;
}

double derror(double x, double s, double y)
{
	return 2*(x-y)*dphi(s);
}


class Neuron {
public:
	// Constructor.
	Neuron() : s(0), x(1), delta(0) {}

	// Set values according tot input.
	void setVals(double input)
		{ s = input; x = phi(input);}

	// accessor functions
	double get_x() const { return x;}
	double get_s() const { return s;}
	void set_delta(double d) { delta = d;}
	double get_delta() const { return delta;}
private:
	double x;		// state of the neuron
	double s;		// s=phi(x)
	double delta;	// value of partial deriv.
};


class Net {
public:
	// constructor
	Net(int Ninn, int Noutt, int Nlayerss, int layerSizee);

	// Train the network. Train using maxIt times 
	// the training date. If rand==true train in
	// randon order.
	void train(std::vector<std::vector<double> >&,
			std::vector<std::vector<double> >&,
			double eta,int maxIt, bool rand=true);

	// Same as above, but stop if descrease in error
	//  in the last two iterations is smaller than
	// tol (or if maxIt times the training data).
	void train(std::vector<std::vector<double> >&,
			std::vector<std::vector<double> >&,
			std::vector<std::vector<double> >&,
			std::vector<std::vector<double> >&,
			double eta, double tol, int maxIt, bool rand=true);

	// Feedforward input and get result in output
	// Both are vectors of vectors with size Nin, Nout
	// respectively.
	void result(const std::vector<std::vector<double> >& input,
			std::vector<std::vector<double> >& output);
	// Feedforward single input, result in saved in output.
	void result(const std::vector<double> & input, std::vector<double>& output);

private:
	int Nin;		// input dimension
	int Nout;		// output dimension
	int Nlayers;	// # of layers(>2)
	int layerSize;	// size of hidden layers

	// feedforward(input,output)
	void feedForward(const std::vector<double>&, std::vector<double>& );
	// feedforward(input)
	void feedForward(const std::vector<double>&);
	// feedback(target)
	void feedBack(const std::vector<double>&);
	// update weights: newW(eta)
	void newW(double);

	// network of neurons
	typedef std::vector<Neuron> Layer;
	std::vector<Layer> network;

	// connections: [layer][from neuron][to neuron]
	std::vector<std::vector<std::vector<double> > > c;

	// random number to initialize connection strength
//	double random(double r) { return 2*(0.5-rand()/double(RAND_MAX))/r;}
	double random(double r) { return 1./r;}
};

void Net::feedBack(const std::vector<double>& target)
{

	assert(target.size() == Nout);

	// set deltas in last layer
	for(int n=0;n<target.size();++n) {
		Neuron& neuron = network.back()[n];
		neuron.set_delta( derror(neuron.get_x(), neuron.get_s(), target[n]) ); 
	}

	for(int l=network.size()-2;l>=0;--l) {
		for(int n=0;n<network[l].size();++n) {
			double delta =0;
			for(int nn=0;nn<network[l+1].size();++nn) {
				delta += c[l][n][nn] * network[l+1][nn].get_delta();
			}
			network[l][n].set_delta(delta*dphi(network[l][n].get_s()));
		}
	}
	
}

void Net::newW(double eta)
{
	for(int l=0;l<(network.size()-2);++l) {
		for(int i=0;i<c[l].size();++i) {
			for(int j=0;j<c[l+1].size()-1;++j) {
				c[l][i][j] -= eta*network[l][i].get_x()*network[l+1][j].get_delta();
			}
		}
	}

}



void Net::feedForward(const std::vector<double>& input, std::vector<double>& output)
{
	assert(input.size() == Nin);
	assert(output.size() == Nout); 

	//set input
	// -1 because of bias neuron
	for(int n=0;n<network[0].size()-1;++n) network[0][n].setVals(input[n]);
	// start feedForward
	for(int l=1;l<network.size();++l) {

		int maxN = l == network.size() -1 ? network[l].size() : network[l].size() -1;
		for(int n=0;n<maxN;++n) {
			double sum =0;
	
			for(int prevn=0;prevn<network[l-1].size();++prevn) {
				sum += c[l-1][prevn][n]*network[l-1][prevn].get_x();
			}
			network[l][n].setVals(sum);
		}
	}

	// copy result in ouput
	for(int n=0;n<network.back().size();++n) output[n] = network.back()[n].get_x();

}

void Net::feedForward(const std::vector<double>& input)
{
	assert(input.size() == Nin);

	//set input
	// -1 because of bias neuron
	for(int n=0;n<network[0].size()-1;++n) network[0][n].setVals(input[n]);
	// start feedForward
	for(int l=1;l<network.size();++l) {

		int maxN = l == network.size() -1 ? network[l].size() : network[l].size() -1;
		for(int n=0;n<maxN;++n) {
			double sum =0;
	
			for(int prevn=0;prevn<network[l-1].size();++prevn) {
				sum += c[l-1][prevn][n]*network[l-1][prevn].get_x();
			}
			network[l][n].setVals(sum);
		}
	}

}


Net::Net(int Ninn, int Noutt, int Nlayerss, int layerSizee)
	: Nin(Ninn), Nout(Noutt), Nlayers(Nlayerss), layerSize(layerSizee)
{
	assert(Nlayers >=2);

	c = std::vector<std::vector<std::vector<double> > >(Nlayers-1);

	network = std::vector<Layer>(Nlayers);
	int ls =0;		// 'this' layer size
	int lsn = 0;	// next layer size
	for(int l=0;(l<Nlayers-1);++l) {
		if (l==0) ls = Nin;
		else ls = layerSize;
		if (l==Nlayers-1) lsn = Nout;
		else lsn = layerSize;

		c[l] = std::vector<std::vector<double> >(ls+1);
		network[l] = Layer(ls+1);
	
		for(int n=0;n<(ls+1);++n) {
			c[l][n] = std::vector<double>(lsn);
			for(int nn =0;nn<c[l][n].size();++nn) {
				c[l][n][nn] = random(lsn);
			}
		}
	}

	network.back() = Layer(Nout);
	
}
void Net::train(std::vector<std::vector<double> >& input,
		std::vector<std::vector<double> >& target,
		double eta, int maxIt, bool randOrder)
{
	int Ntrain = input.size();
	assert(Ntrain==target.size());

	for(int it=0;it<maxIt;++it) {
		// train epoch
		int n = 0;
		for(int ni=0;ni<Ntrain;++ni) {
			if(randOrder) n = rand()%Ntrain;
			feedForward(input[n]);
			feedBack(target[n]);
			newW(eta);
			++n;
		}
	}
}

void Net::result(const std::vector<double> & input, std::vector<double>& output)
{
	feedForward(input,output);

}

void Net::result(const std::vector<std::vector<double> >& input,
		std::vector<std::vector<double> >& output)
{
	int N = input.size();
	assert(N==output.size());

	for(int i=0;i<N;++i)
		feedForward(input[i],output[i]);
}

void Net::train(std::vector<std::vector<double> >& input,
		std::vector<std::vector<double> >& target,
		std::vector<std::vector<double> >& inputTest,
		std::vector<std::vector<double> >& targetTest,
		double eta, double tol, int maxIt, bool randOrder)
{
	int Ntrain = input.size();
	assert(Ntrain==target.size());
	int Ntest = inputTest.size();
	assert(Ntest==targetTest.size());

	std::vector<std::vector<double> > outputTest(targetTest);

	double e1 = tol + 1;
	double e2 = tol + 1;
	double e3 = tol + 1;
	int n = 0;
	std::vector<double> out(Nout);
	int it = 0;
	do {
		// train epoch
		for(int ni=0;ni<Ntrain;++ni) {
			if(randOrder) n = rand();
			else ++n;
			n=n%Ntrain;
			feedForward(input[n],out);
			feedBack(target[n]);
			newW(eta);
		}
		// get test results
		result(inputTest,outputTest);

		// calculate error
		if (tol!=0.0){
			e1 = e2;
			e3 = 0.0;
			for(int i=0;i<Ntest;++i) {
				for(int j=0;j<Nout;++j)
					e2 += (outputTest[i][j]-targetTest[i][j])*(outputTest[i][j]-targetTest[i][j]);
			}
			e3 /= Nout;
			e3 = sqrt(e3);
			e2 = abs(e2-e3);
			if(e1<tol and e2<tol) break;
		}	
	
		++it;
	}while(it<maxIt);

}

#endif

