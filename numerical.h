/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   numerical.h
 * Author: ziqi
 *
 * Created on January 29, 2019, 7:31 PM
 */

#ifndef NUMERICAL_H
#define NUMERICAL_H

#include <iostream>
#include <vector>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <gsl/gsl_sf.h>
//#include <gsl/gsl_math.h>
//#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
using namespace std;

#ifndef IDXC0
#define IDXC0(row,col,ld) ((ld)*(col)+(row))
#endif

#ifndef max
#define max(a,b) (a > b ? a : b)
#endif

#ifndef min
#define min(a,b) (a < b ? a : b)
#endif

#ifndef HOST_CALL
#define HOST_CALL(x) do {\
if(x!=EXIT_SUCCESS){\
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)
#endif

//class gaussQuad
class gaussQuad {
    friend ostream& operator<<(ostream&, const gaussQuad&);
private:
    float *evalPnts = NULL;
    float *wgts = NULL;
    int n; //order of integration
    
    int genGaussParams();
    

    
public:
    gaussQuad(): n(7) { evalPnts=new float[n]; wgts=new float[n]; genGaussParams();}
    gaussQuad(const int);
    gaussQuad(const gaussQuad&);
    ~gaussQuad();
    
    gaussQuad& operator=(const gaussQuad&);
    
    
};

//class point
class point {
    friend ostream& operator<<(ostream&,const point&);
private:
    float coords[3];

public:
    point() {coords[0]=0;coords[1]=0;coords[2]=0;}
    point(const point&);
    point& operator=(const point&);
    ~point() = default;
    void set(const float,const float,const float);
};

ostream& operator<<(ostream&,const point&);

//class triElem
class triElem {
private:
    int nodes[3];
    cuFloatComplex bc[3];
    
public:
    triElem() = default;
    triElem(const triElem&);
    ~triElem() = default;
    triElem& operator=(const triElem&);
};

#endif /* NUMERICAL_H */

