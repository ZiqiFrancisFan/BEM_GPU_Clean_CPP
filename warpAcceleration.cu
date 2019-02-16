/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "warpAcceleration.h"

__global__ void test_shfl_broadcast(float *d_out, float *d_in, const int srcLane) {
    float value = d_in[threadIdx.x];
    value = __shfl_sync(0xffffffff,value,srcLane,BDIMX);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_up(float *d_out, float *d_in, const int delta) {
    float value = d_in[threadIdx.x];
    value = __shfl_up_sync(0xffffffff,value,delta,16);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_down(float *d_out, float *d_in, const int delta) {
    float value = d_in[threadIdx.x];
    value = __shfl_down_sync(0xffffffff,value,delta,16);
    d_out[threadIdx.x] = value;
}

__global__ void test_shfl_xor(float *d_out, float *d_in, const int mask) {
    float value = d_in[threadIdx.x];
    value = __shfl_xor_sync(0xffffffff,value,mask,BDIMX);
    d_out[threadIdx.x] = value;
}

__inline__ __device__ float warpReduce(float mySum) {
    mySum+=__shfl_xor_sync(0xffffffff,mySum,16);
    mySum+=__shfl_xor_sync(0xffffffff,mySum,8);
    mySum+=__shfl_xor_sync(0xffffffff,mySum,4);
    mySum+=__shfl_xor_sync(0xffffffff,mySum,2);
    mySum+=__shfl_xor_sync(0xffffffff,mySum,1);
    return mySum;
}

__global__ void reduceShfl(int *g_idata, int *g_odata, unsigned int n) {
    // shared memory for each warp sum
    __shared__ int smem[SMEMDIM];
    // boundary check
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<n) {
        // read from global memory
        int mySum = g_idata[idx];
        // calculate lane index and warp index
        int laneIdx = threadIdx.x%warpSize;
        int warpIdx = threadIdx.x/warpSize;
        // block-wide warp reduce
        mySum = warpReduce(mySum);
        // save warp sum to shared memory
        if(laneIdx==0) smem[warpIdx] = mySum;
        // block synchronization
        __syncthreads();
        // last warp reduce
        mySum = (threadIdx.x<SMEMDIM) ? smem[laneIdx] : 0;
        if(warpIdx==0) mySum = warpReduce(mySum);
        // write result for this block to global mem
        if(threadIdx.x==0) g_odata[blockIdx.x] = mySum;
    }
}
