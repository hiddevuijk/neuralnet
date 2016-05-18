#ifndef GUARD_net_h
#define GUARD_net_h


#include <math.h>
#include <vector>
#include <stdlib.h>
#include <iostream>

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
	Neuron() : s(0), x(1), delta(0) {}

	void setVals(double input)
		{ s = input; x = phi(input);}

	double get_x() const { return x;}
	double get_s() const { return s;}
	void set_delta(double d) { delta = d;}
	double get_delta() const { return delta;}
private:
	double s,x,delta;

};


class Net {
public:
	Net(const std::vector<int>&);

	void feedForward(const std::vector<double>&, std::vector<double>& );
	void feedBack(const std::vector<double>&);
	void newW(double);
	void feedBack_newW(const std::vector<double>&, double);
private:
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

	// assert

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

void Net::feedBack_newW(const std::vector<double>& target, double eta)
{


}

void Net::feedForward(const std::vector<double>& input, std::vector<double>& output)
{
	// assert 

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


Net::Net(const std::vector<int>& layerSizes)
{
	c = std::vector<std::vector<std::vector<double> > >(layerSizes.size()-1);

	network = std::vector<Layer>(layerSizes.size());

	for(int l=0;(l<layerSizes.size()-1);++l) {
		c[l] = std::vector<std::vector<double> >(layerSizes[l]+1);
		network[l] = Layer(layerSizes[l]+1);

		for(int n=0;n<(layerSizes[l]+1);++n) {
			c[l][n] = std::vector<double>(layerSizes[l+1]);
			for(int nn =0;nn<c[l][n].size();++nn) {
				c[l][n][nn] = random(layerSizes[l+1]);
			}
		}
	}

	network.back() = Layer(layerSizes.back());
	
}

class NeuralNetwork {
public:
	NeuralNetwork(int Nin, int Nout, int Nlayers, int layerSize);
	
	void train(const std::vector<std::vector<double> >& inputs,
		const std::vector<std::vector<double> >& targets, 
		double eta, double tol, int maxIter, int seed);

	void getOutput(const std::vector<double>& input, std::vector<double>& output);
	void getOutput(const std::vector<std::vector<double> >& inputs,
			std::vector<std::vector<double> >& ouputs);
	double error() const { return er; }

	double score(const std::vector<std::vector<double> >& input, 
			const std::vector<std::vector<double> >& target);
	void reset(int seed);

private:
	Net n;
	double er; 
};




#endif

