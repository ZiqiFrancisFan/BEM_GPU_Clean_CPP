/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: ziqi
 *
 * Created on January 29, 2019, 7:28 PM
 */

#include <cstdlib>
#include <time.h>
#include "numerical.h"
#include "mesh.h"
#include <cuda.h>
#include "device_launch_parameters.h"

using namespace std;

int main(int argc, char** argv) {
    gaussQuad gss(INTORDER);
    gss.sendToDevice();
    float f = 25;
    float k = 2*PI*f/343.21;
    mesh m;
    m.readObj("KEMARTORSO_12000Hz.obj");
    m.findBB(0.01);
    m.genCHIEF(1,0.000001);
    m.printCHIEF();
    
    cartCoord src(3,2,2);
    int numSrcs = 1;
    
    cuFloatComplex *A = new cuFloatComplex[(m.getNumPnts()+m.getNumChief())
            *m.getNumPnts()*sizeof(cuFloatComplex)];
    cuFloatComplex *B = new cuFloatComplex[(m.getNumPnts()+m.getNumChief())*numSrcs
            *sizeof(cuFloatComplex)];
    clock_t t;
    t = clock();
    bemSystem(m,k,&src,1,A,(m.getNumPnts()+m.getNumChief()),B,(m.getNumPnts()+m.getNumChief()));
    t = clock()-t;
    printf("Elapsed %f seconds in generation of system.\n",((float)t)/CLOCKS_PER_SEC);
    
    
    
    delete[] A;
    delete[] B;
    return 0;
}

