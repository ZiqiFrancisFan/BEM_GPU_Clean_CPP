/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   core.h
 * Author: ziqi
 *
 * Created on May 6, 2018, 8:15 PM
 */

#ifndef GMRES_H
#define GMRES_H


#ifndef IDXC0
#define IDXC0(row,column,stride) ((column)*(stride)+(row))
#endif

#ifndef IDXC1
#define IDXC1(row,column,stride) ((column-1)*stride+(row-1))
#endif

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#ifndef max
#define max(a,b) (a>b?a:b)
#endif

#ifndef min
#define min(a,b) (a<b?a:b)
#endif

#ifndef CUDA_CALL
#define CUDA_CALL(x) \
do { \
    if((x)!=cudaSuccess) \
    { printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE; } \
} \
while(0)
#endif

#ifndef CURAND_CALL
#define CURAND_CALL(x) \
do { \
    if((x)!=CURAND_STATUS_SUCCESS) \
    { printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE; } \
} \
while(0)
#endif

#ifndef CUBLAS_CALL
#define CUBLAS_CALL(x) \
do { \
    if((x)!=CUBLAS_STATUS_SUCCESS) \
        { \
            printf("Error at %s:%d\n",__FILE__,__LINE__); \
            if(x==CUBLAS_STATUS_NOT_INITIALIZED) { \
                printf("The library was not initialized.\n"); \
            } \
            if(x==CUBLAS_STATUS_INVALID_VALUE) { \
                printf("There were problems with the parameters.\n"); \
            } \
            if(x==CUBLAS_STATUS_MAPPING_ERROR) { \
                printf("There was an error accessing GPU memory.\n"); \
        } \
            return EXIT_FAILURE; } \
    } \
while(0)
#endif

#ifndef HOST_CALL
#define HOST_CALL(x) \
do {\
    if(x!=EXIT_SUCCESS){ \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;} \
} while(0)
#endif


extern "C" HOST void Rsolver(const cuFloatComplex *R, const cuFloatComplex *b, 
        const int m, cuFloatComplex *x);

extern "C" HOST void printMatrix(cuFloatComplex *A, const int row, const int col, 
        const int lda);

extern "C" HOST void givens_coeffs(const cuFloatComplex rho, 
        const cuFloatComplex sigma, cuFloatComplex *c, cuFloatComplex *s);

extern "C" HOST int arnoldi(const cuFloatComplex *A_h, const int k, const int m, 
        cuFloatComplex *Q_h, cuFloatComplex *H_h);

int GMRES(const cuFloatComplex *A_h, const cuFloatComplex *b_h, const int m, 
        const int maxIter, const float threshold, cuFloatComplex *x_h);
#endif /* GMRES_H */

