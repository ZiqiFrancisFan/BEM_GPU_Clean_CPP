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

using namespace std;

int main(int argc, char** argv) {
    gaussQuad g(7);
    cout << g << endl;
    
    
    return 0;
}

