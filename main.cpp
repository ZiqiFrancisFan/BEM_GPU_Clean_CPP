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
#include "GMRES.h"
#include <cuda.h>
#include "device_launch_parameters.h"

using namespace std;

int main(int argc, char** argv) {
    
    CUDA_CALL(cudaDeviceReset());
    
    float f = 171.5;
    float k = 2*PI*f/343.21;
    mesh m;
    size_t fr, ttl;
    m.readObj("Head_20kHz.obj");
    m.findBB(0.0001);
    m.genCHIEF(5,0.0001);
    std::cout << "CHIEF points generated." << std::endl;
    m.printCHIEF();
    gaussQuad gss(INTORDER);
    gss.sendToDevice();
    std::cout << "Integral sent to device." << std::endl;
    
    
    cartCoord src(1000000,1000000,1000000);
    int numSrcs = 1;
    std::cout << "Allocating memory for matrices A and B" << std::endl;
    cuFloatComplex *A = new cuFloatComplex[(m.getNumPnts()+m.getNumChief())
            *m.getNumPnts()*sizeof(cuFloatComplex)];
    
    cuFloatComplex *B = new cuFloatComplex[(m.getNumPnts()+m.getNumChief())*numSrcs
            *sizeof(cuFloatComplex)];
    std::cout << "Completed allocating memory." << std::endl;
    
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
    //CUDA_CALL(cudaDeviceSynchronize());
    
    cuFloatComplex *Q = new cuFloatComplex[(m.getNumPnts()+m.getNumChief())
            *(m.getNumPnts()+m.getNumChief())];
    t = clock();
    HOST_CALL(lsqSolver(A,m.getNumPnts()+m.getNumChief(),m.getNumPnts(),
            m.getNumPnts()+m.getNumChief(),B,numSrcs,m.getNumPnts()+m.getNumChief(),Q));
    t = clock()-t;
    printf("Elapsed %f seconds in solution of system.\n",((float)t)/CLOCKS_PER_SEC);
    float radius = 3;
    float step = 0.2;
    int numLocs = (radius-1)/step+1;
    cuFloatComplex *pressure = new cuFloatComplex[numLocs];
    pressure[0] = B[0];
    
    
    //printComplexMatrix(B,m.getNumPnts(),numSrcs,m.getNumPnts()+m.getNumChief());
    
    /*
    cuFloatComplex *x = new cuFloatComplex[m.getNumPnts()];
    GMRES(A,B,m.getNumPnts(),m.getNumPnts(),0,x);
    printComplexMatrix(x,m.getNumPnts(),1,m.getNumPnts());
    float radius = 3;
    float step = 0.2;
    int numLocs = (radius-1)/step+1;
    cuFloatComplex *pressure = new cuFloatComplex[numLocs];
    pressure[0] = x[0];
     */ 
    for(int i=1;i<numLocs;i++) {
        pressure[i] = genExtPressure(k,m,src,cartCoord(-(1+step*i),0,0),B);
    }
    //printComplexMatrix(pressure,1,numLocs,1);
    //wrtCplxMat(pressure,1,numLocs,1,"sphere171Hz_CHIEF");
    std::cout << cmptSurfArea(m) << std::endl;
    
    delete[] A;
    delete[] B;
    //delete[] Q;
    delete[] pressure;
    //delete[] x;
    return 0;
}

