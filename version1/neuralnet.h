#ifndef GUARD_neuralnet_h
#define GUARD_neuralnet_h

#include <math.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

class Net;
class Neuron;

//default input-output functions of the neurons
double phi(double a) { return tanh(a);}
double dphi(double a) { return 1-tanh(a)*tanh(a);}


class Neuron {
public:
	// constructor
	Neuron() : s(1.), x(0), delta(0) {}

	// Set values according to input
	// to that neuron
	void setVals(double input)
		{x = input; s = f(input);}

	// set delta
	void set_delta(double d) { delta = d;}

	// accessor functions
	double get_x() const { return x;}
	double get_s() const { return s;}
	double get_delta() const { return delta;}

	// set firing rate functions
	void setF(double(*ff)(double),double(*dff)(double))
			{ f = ff; df=dff;}

private:
	double x;		// state of the neuron
	double s;		// s=f(x)
	double delta;	// value of partial deriv.
public:	
	double (*f)(double);
	double (*df)(double);
};


class Net{
public:
	// constructor
//	Net(unsigned int,unsigned int,unsigned int,unsigned int);
	Net(unsigned int,unsigned int,unsigned int,unsigned int,
		double(*ff)(double)=phi, double (*dff)(double)=dphi );
	//feedForward
	void feedForward(const std::vector<double>& input,std::vector<double>& output);

	// train single input-target
	void train(const std::vector<double>& input,
			const std::vector<double>& target,double eta);

	// train vector with input. maxTrain times training data.
	// if r=false in order
	// if r=true train maxTrain*inputVecs.size() times a random input-target pair
	void train(const std::vector<std::vector<double> >& inputVecs,
			const std::vector<std::vector<double> >& targetVecs,
			int maxTrain,double eta, bool r=false);
	

private:
	unsigned int Nin;		// input dim.
	unsigned int Nout;		// output dim.
	unsigned int Nlayer;	// # of hidden layers
	unsigned int layerSize;	// # neuron in hidden layers

	// network of neurons
	std::vector<std::vector<Neuron> > network;

	// connections
	std::vector<std::vector<std::vector<double> > > c;

	double cInit(double n) { return (0.5-rand()/double(RAND_MAX))/n;}

	//feedForward
	void feedForward(const std::vector<double>& input);

	// feedBack
	void feedBack(const std::vector<double>& target);

	// update connections
	void newC(double eta);

	// transfer function
	double (*f)(double);
	double (*df)(double);

	// error functions
	double error(const std::vector<double>&, const std::vector<double>&);
	double derror(double,double,double);


};
void Net::train(const std::vector<std::vector<double> >& inputVecs,
			const std::vector<std::vector<double> >& targetVecs,
			int maxTrain,double eta, bool r)
{
	int Nv = inputVecs.size();
	assert(Nv==targetVecs.size());

	int i;
	for(int nTrain=0;nTrain<maxTrain;++nTrain) {
		for(int nv=0;nv<Nv;++nv) {
			i = r ? rand()%Nv : nv;
			train(inputVecs[i],targetVecs[i],eta);
		}
	} 
}
	

void Net::train(const std::vector<double>& input,
			const std::vector<double>& target,double eta)
{
	feedForward(input);
	feedBack(target);
	newC(eta);	

}


void Net::feedBack(const std::vector<double>& target)
{
	assert(target.size() == Nout);

	// set delta's in last layer
	for(int n=0;n<Nout;++n) {
		// reference to neuron n in output layer
		Neuron& neuron = network.back()[n];
		neuron.set_delta(derror(neuron.get_s(),neuron.get_x(),target[n]));
	}

	// start with last hidden layer
	for(int l=Nlayer;l>=0;--l) {
		// n = neuron index in layer l
		for(int n=0;n<network[l].size()-1;++n) {
			double delta=0;
			for(int nn=0;nn<network[l+1].size()-1;++nn) {
				delta += c[l][n][nn]*network[l+1][nn].get_delta();
			}
			network[l][n].set_delta(delta*dphi(network[l][n].get_s()));
		}
	}
}

void Net::newC(double eta)
{
	for(int l=0;l<c.size();++l) {
		for(int i=0;i<c[l].size();++i) {
			// check j index in network
			int jMax = l==c.size()-1 ? network[l+1].size() : network[l+1].size()-1;
			for(int j=0;j<jMax;++j) {
				c[l][i][j] -= eta*network[l][i].get_x()*network[l+1][j].get_delta();
			}
		}
	}

}


void Net::feedForward(const std::vector<double>& input, std::vector<double>& output)
{
	assert(output.size() == Nout);
	assert(input.size() == Nin);
	// feedforward
	feedForward(input);

	// copy outputlayer in output
	for(int n=0;n<Nout;++n)
		output[n] = network.back()[n].get_s();
}

void Net::feedForward(const std::vector<double>& input)
{
	// set input
	for(int n=0;n<(network[0].size()-1);++n)
		network[0][n].setVals(input[n]);

	// start feedforward
	for(int l=1;l<network.size();++l) {
		// n = index of neuron in layer l
		int nMax = l==network.size()-1? network[l].size() : network[l].size()-1;
		for(int n=0;n<nMax;++n) {
			double sum =0;
			// prevn = index of neuron in prev. layer
			for(int prevn=0;prevn<network[l-1].size();++prevn) {
				sum += c[l-1][prevn][n]*network[l-1][prevn].get_s();
			}
			network[l][n].setVals(sum);
		}
	}
}


	
Net::Net(unsigned int Ninn,unsigned  int Noutt,
		unsigned  int Nlayerr,unsigned int layerSizee,
		double(*ff)(double) , double(*dff)(double))
	: Nin(Ninn), Nout(Noutt), Nlayer(Nlayerr), layerSize(layerSizee),
	f(ff),df(dff)
{
	assert(Nin>0);
	assert(Nout>0);
	assert(Nlayer>=0);
	// if there are hidden layers they must have nonzero size
	if(Nlayer > 0) assert(layerSize>0);

	// connections
	c = std::vector<std::vector<std::vector<double> > >(1+Nlayer);

	// network of neurons. input + output layer = 2
	network = std::vector<std::vector<Neuron> >(2+Nlayer);


	//initialize input layer
	// add a bias neuron
	network[0] = std::vector<Neuron>(Nin+1);

	// if Nlayer> 0, initialize hidden layers
	// l = hidden layer index
	if(Nlayer>0){
		for(int l=1;l<(Nlayer+1);++l) {
			network[l] = std::vector<Neuron>(layerSize+1);
		}
	}

	// initialize output layer
	network[Nlayer+1] = std::vector<Neuron>(Nout);

	// set firing rate functions
	for(int i=0;i<network.size();++i) {
		for(int j=0;j<network[i].size();++j) {
			network[i][j].setF(f,df);
		}
	}	
	// Initialize connections

	int ls;		// size of 'from' layer
	int lsn;	// size of 'to' layer
	for(int l=0;l<(Nlayer+1);++l) {
		ls = l==0 ? Nin : layerSize;
		lsn = l==Nlayer ? Nout : layerSize;

		// ls + 1 b.c. bias neuron
		c[l] = std::vector<std::vector<double> >(ls+1);
		// n is 'from' neuron index
		for(int n=0;n<(ls+1);++n) {
			c[l][n] = std::vector<double>(lsn);
			// nn if 'to' neuron index
			for(int nn=0;nn<lsn;++nn) {
				c[l][n][nn] = cInit(lsn);
			}
		}
	}
}


double Net::error(const std::vector<double>& output, const std::vector<double>& target)
{
	assert(output.size()==target.size());
	double e = 0.;
	for(int n=0;n<output.size();++n)
		e += (output[n]-target[n])*(output[n]-target[n]);
	return e;
}


double Net::derror(double s,double x,double y)
{
	return 2*(s-y)*df(x);
}




#endif
