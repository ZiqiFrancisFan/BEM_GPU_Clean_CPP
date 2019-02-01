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

#include "numerical.h"
#include "mesh.h"
#include <cuda.h>
#include "device_launch_parameters.h"

using namespace std;

int main(int argc, char** argv) {
    //Test();
    cartCoord p1(1,0,0), p2(1,-1,0), p3(2,1,0), pnt = p1+numMul(0.2,(p2-p1))+numMul(0.1,(p3-p2));
    cout << pnt << endl;
    cartCoord sp(0.2,0.2,-1), dir = pnt-sp;
    cout << rayTrnglInt(sp,dir,p1,p2,p3) << endl;
    return 0;
}

