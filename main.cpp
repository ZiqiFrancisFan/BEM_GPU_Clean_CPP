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
    cartCoord c1(1,0,0), c2(0,1,0), c3(0,0,1);
    cartCoord2D c(1,0);
    cout << tf2DTo3D(c1,c2,c3,c) << endl;
    
    return 0;
}

