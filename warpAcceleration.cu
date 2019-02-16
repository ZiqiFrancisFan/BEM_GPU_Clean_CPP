/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "warpAcceleration.h"

__global__ void test_shfl_broadcast(float *d_out, float *d_in, const int srcLane) {
    float value = d_in[threadIdx.x];
    value = __shfl(value,srcLane,16);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_up(float *d_out, float *d_in, const int delta) {
    float value = d_in[threadIdx.x];
    value = __shfl_up(value,delta,16);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_down(float *d_out, float *d_in, const int delta) {
    float value = d_in[threadIdx.x];
    value = __shfl_down(value,delta,16);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_xor(float *d_out, float *d_in, const int mask) {
    float value = d_in[threadIdx.x];
    value = __shfl_xor(value,mask,BDIMX);
    d_out[threadIdx.x] = value;
}

__inline__ __device__ float warpReduce(float mySum) {
    mySum+=__shfl_xor(mySum,16);
    mySum+=__shfl_xor(mySum,8);
    mySum+=__shfl_xor(mySum,4);
    mySum+=__shfl_xor(mySum,2);
    mySum+=__shfl_xor(mySum,1);
    return mySum;
}
