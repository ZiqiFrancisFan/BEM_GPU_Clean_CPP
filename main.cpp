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
    float f = 171.5;
    float k = 2*PI*f/343.21;
    mesh m;
    size_t fr, ttl;
    m.readObj("sphere1.obj");
    m.findBB(0.01);
    m.genCHIEF(30,0.001);
    m.printCHIEF();
    
    
    cartCoord src(10000,10000,10000);
    int numSrcs = 1;
    
    cuFloatComplex *A = new cuFloatComplex[(m.getNumPnts()+m.getNumChief())
            *m.getNumPnts()*sizeof(cuFloatComplex)];
    cuFloatComplex *B = new cuFloatComplex[(m.getNumPnts()+m.getNumChief())*numSrcs
            *sizeof(cuFloatComplex)];
    
    
    CUDA_CALL(cudaMemGetInfo(&fr,&ttl));
    printf("free GPU memory: %f,total: %f, before the program starts\n",
            (float)fr/1024.0/1024.0/1024.0,(float)ttl/1024.0/1024.0/1024.0);
    clock_t t;
    t = clock();
    HOST_CALL(bemSystem(m,k,&src,numSrcs,A,(m.getNumPnts()+m.getNumChief()),B,
            (m.getNumPnts()+m.getNumChief())));
    t = clock()-t;
    //printComplexMatrix(A,m.getNumPnts()+m.getNumChief(),m.getNumPnts(),m.getNumPnts()+m.getNumChief());
    printf("Elapsed %f seconds in generation of system.\n",((float)t)/CLOCKS_PER_SEC);
    CUDA_CALL(cudaDeviceSynchronize());
    cuFloatComplex *Q = new cuFloatComplex[(m.getNumPnts()+m.getNumChief())
            *(m.getNumPnts()+m.getNumChief())];
    HOST_CALL(lsqSolver(A,m.getNumPnts()+m.getNumChief(),m.getNumPnts(),
            m.getNumPnts()+m.getNumChief(),B,numSrcs,m.getNumPnts()+m.getNumChief(),Q));
    
    printComplexMatrix(B,m.getNumPnts(),numSrcs,m.getNumPnts()+m.getNumChief());
    
    
    delete[] A;
    delete[] B;
    delete[] Q;
    return 0;
}

