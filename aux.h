#ifndef GUARD_aux_h
#define GUARD_aux_h

#include <vector>
#include <math.h>

void rescale(std::vector<std::vector<double> >& vv)
{

	double avg = 0.0;
	double largest = 0.0;
	unsigned long int nelem = 0;
	for(int Ni=0;Ni<vv.size();++Ni) {
		nelem += vv[Ni].size();
		for(int ni=0;ni<vv[Ni].size();++ni) {
			avg += vv[Ni][ni];
			if(fabs(vv[Ni][ni]) > largest)
				largest = fabs(vv[Ni][ni]);
		}
	}

	avg /= nelem;

	for(int Ni=0;Ni<N;++Ni) {
		for(int ni=0;ni<n;++ni) {
			vv[Ni][ni] = 


}	




#endif
