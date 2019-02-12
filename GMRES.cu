#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <cuComplex.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "GMRES.h"

extern "C" HOST void printMatrix(cuFloatComplex *A, const int row, const int col, 
        const int lda)
{
	float x, y;
	int i, j;
	for (i=0;i<row;i++) {
            for (j=0;j<col;j++) {
                x = cuCrealf(A[IDXC0(i,j,lda)]);
                y = cuCimagf(A[IDXC0(i,j,lda)]);
                printf("(%f,%f) ",x,y);
            }
            printf("\n");
	}
		
}

extern "C" HOST void Rsolver(const cuFloatComplex *R, const cuFloatComplex *b, 
        const int m, cuFloatComplex *x) 
{
    int i, j;
    cuFloatComplex temp[6];
    for(i=m-1;i>=0;i--) {
        temp[0] = b[i];
        temp[1] = R[IDXC0(i,i,m)];
        for(j=m-1;j>i;j--) {
            temp[2] = x[j];
            temp[3] = R[IDXC0(i,j,m)];
            temp[4] = cuCmulf(temp[2],temp[3]);
            temp[0] = cuCsubf(temp[0],temp[4]);
        }
        x[i] = cuCdivf(temp[0],temp[1]);
    }
}

extern "C" HOST void givens_coeffs(const cuFloatComplex rho, 
        const cuFloatComplex sigma, cuFloatComplex *c, cuFloatComplex *s) 
{
    cuFloatComplex rho_b = cuConjf(rho), sigma_b = cuConjf(sigma);
    float x, y, mag;
    mag = sqrt(pow(cuCabsf(rho),2)+pow(cuCabsf(sigma),2));
    x = cuCrealf(rho_b)/mag;
    y = cuCimagf(rho_b)/mag;
    *c = make_cuFloatComplex(x,y);
    x = cuCrealf(sigma_b)/mag;
    y = cuCimagf(sigma_b)/mag;
    *s = make_cuFloatComplex(x,y);
}

extern "C" HOST void apply_givens(const int m, const int k, cuFloatComplex *c, 
        cuFloatComplex *s, cuFloatComplex *h) 
{
    cuFloatComplex c_k, s_k, c_b, s_b, temp[7];
    float x, y;
    int i;
    for(i=0;i<k-1;i++) {
        temp[0] = cuCmulf(c[i],h[i]);
        temp[1] = cuCmulf(s[i],h[i+1]);
        temp[2] = cuCaddf(temp[0],temp[1]);
        c_b = cuConjf(c[i]);
        s_b = cuConjf(s[i]);
        x = cuCrealf(s_b);
        y = cuCimagf(s_b);
        temp[3] = make_cuFloatComplex(-x,-y);
        temp[4] = cuCmulf(temp[3],h[i]);
        temp[5] = cuCmulf(c_b,h[i+1]);
        temp[6] = cuCaddf(temp[4],temp[5]);
        h[i+1] = temp[6];
        h[i] = temp[2];
    }
    if(k<m) {
        givens_coeffs(h[k-1],h[k],&c_k,&s_k);
        c[k-1] = c_k;
        s[k-1] = s_k;
        temp[0] = cuCmulf(c_k,h[k-1]);
        temp[1] = cuCmulf(s_k,h[k]);
        h[k-1] = cuCaddf(temp[0],temp[1]);
        h[k] = make_cuFloatComplex(0,0);
    }
}

extern "C" HOST int arnoldi(const cuFloatComplex *A_h, const int k, const int m, 
        cuFloatComplex *Q_h, cuFloatComplex *H_h) 
{
    if(k>m) {
        printf("Error with input k.\n");
        return EXIT_FAILURE;
    }
    float v_real, v_imag;
    int i;
    float nrm;
    cuFloatComplex alpha, beta, prod;
    alpha = make_cuFloatComplex(1,0);
    beta = make_cuFloatComplex(0,0);
    cuFloatComplex *A_d, *q_d, *y_d;
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate_v2(&handle));
    CUDA_CALL(cudaMalloc(&A_d,m*m*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(A_d,A_h,m*m*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&q_d,m*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(q_d,&Q_h[IDXC0(0,k-1,m)],m*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&y_d,m*sizeof(cuFloatComplex)));
    CUBLAS_CALL(cublasCgemv_v2(handle,CUBLAS_OP_N,m,m,&alpha,A_d,m,q_d,1,&beta,
            y_d,1)); //Aq
    CUDA_CALL(cudaFree(A_d));
    for(i=1;i<=k;i++) {
        CUDA_CALL(cudaMemcpy(q_d,&Q_h[IDXC0(0,i-1,m)],m*sizeof(cuFloatComplex),
                cudaMemcpyHostToDevice));
        CUBLAS_CALL(cublasCdotc_v2(handle,m,q_d,1,y_d,1,&prod));
        H_h[IDXC0(i-1,k-1,m)] = prod;
        v_real = cuCrealf(prod);
        v_imag = cuCimagf(prod);
        alpha = make_cuFloatComplex(-v_real,-v_imag);
        CUBLAS_CALL(cublasCaxpy_v2(handle,m,&alpha,q_d,1,y_d,1));
    }
    CUBLAS_CALL(cublasScnrm2_v2(handle,m,y_d,1,&nrm));
    if(k<m) {
        H_h[IDXC0(k,k-1,m)] = make_cuFloatComplex(nrm,0);
        alpha = make_cuFloatComplex(1.0/nrm,0);
        CUDA_CALL(cudaMemset(q_d,0,m*sizeof(cuFloatComplex)));
        CUBLAS_CALL(cublasCaxpy_v2(handle,m,&alpha,y_d,1,q_d,1));
        CUDA_CALL(cudaMemcpy(&Q_h[IDXC0(0,k,m)],q_d,m*sizeof(cuFloatComplex),
                cudaMemcpyDeviceToHost));
    }
    //printf("Q in arnoldi: \n");
    //printMatrix(Q_h,m,m,m);
    
    CUDA_CALL(cudaFree(y_d));
    CUDA_CALL(cudaFree(q_d));
    CUBLAS_CALL(cublasDestroy_v2(handle));
    
    return EXIT_SUCCESS;
}

int GMRES(const cuFloatComplex *A_h, const cuFloatComplex *b_h, const int m, 
        const int maxIter, const float threshold, cuFloatComplex *x_h)
{
    //printf("input x: \n");
    //printMatrix(x_h,m,1,m);
    int i, j, t;
    float x, y;
    cublasHandle_t handle;
    cuFloatComplex alpha, beta;
    cuFloatComplex *Q_h = (cuFloatComplex*)malloc(m*m*sizeof(cuFloatComplex));
    cuFloatComplex *H_h = (cuFloatComplex*)malloc(m*m*sizeof(cuFloatComplex));
    cuFloatComplex *c = (cuFloatComplex*)malloc(m*sizeof(cuFloatComplex));
    cuFloatComplex *s = (cuFloatComplex*)malloc(m*sizeof(cuFloatComplex));
    cuFloatComplex *h = (cuFloatComplex*)malloc(m*sizeof(cuFloatComplex));
    cuFloatComplex *err_h = (cuFloatComplex*)malloc(m*sizeof(cuFloatComplex));
    cuFloatComplex *A_d, *x_d, *r_d, *q_d;
    CUDA_CALL(cudaMalloc(&A_d,m*m*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(A_d,A_h,m*m*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&x_d,m*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(x_d,x_h,m*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&r_d,m*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(r_d,b_h,m*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&q_d,m*sizeof(cuFloatComplex)));
    float nrm_b, nrm_r;
    CUBLAS_CALL(cublasCreate_v2(&handle));
    CUBLAS_CALL(cublasScnrm2_v2(handle,m,r_d,1,&nrm_b)); //norm of b vector
    alpha = make_cuFloatComplex(-1,0);
    beta = make_cuFloatComplex(1,0);
    CUBLAS_CALL(cublasCgemv_v2(handle,CUBLAS_OP_N,m,m,&alpha,A_d,m,x_d,1,&beta,
            r_d,1)); //r = b-Ax
    CUDA_CALL(cudaFree(A_d));
    CUBLAS_CALL(cublasScnrm2_v2(handle,m,r_d,1,&nrm_r)); //norm of r vector
    err_h[0] = make_cuFloatComplex(nrm_r,0);
    //printf("nrm_r=%f\n",nrm_r);
    if(nrm_r/nrm_b<threshold) {
        printf("The initial x is accurate enough.\n");
        CUDA_CALL(cudaFree(q_d));
        CUDA_CALL(cudaFree(r_d));
        CUDA_CALL(cudaFree(x_d));
        free(err_h);
        free(h);
        free(s);
        free(c);
        free(H_h);
        free(Q_h);
        return EXIT_SUCCESS;
    }
    CUDA_CALL(cudaMemset(q_d,0,m*sizeof(cuFloatComplex)));
    alpha = make_cuFloatComplex(1.0/nrm_r,0);
    CUBLAS_CALL(cublasCaxpy_v2(handle,m,&alpha,r_d,1,q_d,1));
    CUDA_CALL(cudaMemcpy(&Q_h[IDXC0(0,0,m)],q_d,m*sizeof(cuFloatComplex),
            cudaMemcpyDeviceToHost));
    i = 1;
    while(1) {
        arnoldi(A_h,i,m,Q_h,H_h);
        //printf("H_h before givens: \n");
        //printMatrix(H_h,m,i,m);
        if(i<m) {
            CUDA_CALL(cudaMemcpy(h,&H_h[IDXC0(0,i-1,m)],
                    (i+1)*sizeof(cuFloatComplex),cudaMemcpyHostToHost));
            apply_givens(m,i,c,s,h);
            CUDA_CALL(cudaMemcpy(&H_h[IDXC0(0,i-1,m)],h,
                    (i+1)*sizeof(cuFloatComplex),cudaMemcpyHostToHost));
            alpha = cuConjf(s[i-1]);
            x = cuCrealf(alpha);
            y = cuCimagf(alpha);
            beta = make_cuFloatComplex(-x,-y);
            err_h[i] = cuCmulf(beta,err_h[i-1]);
            err_h[i-1] = cuCmulf(c[i-1],err_h[i-1]);
        } else {
            CUDA_CALL(cudaMemcpy(h,&H_h[IDXC0(0,i-1,m)],m*sizeof(cuFloatComplex),
                    cudaMemcpyHostToHost));
            apply_givens(m,i,c,s,h);
            CUDA_CALL(cudaMemcpy(&H_h[IDXC0(0,i-1,m)],h,m*sizeof(cuFloatComplex),
                    cudaMemcpyHostToHost));
        }
        if((i<m && cuCabsf(err_h[i])/nrm_b<threshold) || i>=maxIter) {
            break;
        }
        i++;
        printf("Iteration: %d\n",i);
    }
    //printf("c: \n");
    //printMatrix(c,m-1,1,m);
    //printf("s: \n");
    //printMatrix(s,m-1,1,m);
    //printf("err_h: \n");
    //printMatrix(err_h,m,1,m);
    t = i;
    cuFloatComplex *y_h = (cuFloatComplex*)malloc(t*sizeof(cuFloatComplex));
    cuFloatComplex *R_h = (cuFloatComplex*)malloc(t*t*sizeof(cuFloatComplex));
    for(i=0;i<t;i++) {
        for(j=0;j<t;j++) {
            R_h[IDXC0(i,j,t)] = H_h[IDXC0(i,j,m)];
        }
    }
    //printf("H_h: \n");
    //printMatrix(H_h,m,m,m);
    //printf("R_h: \n");
    //printMatrix(R_h,i,i,i);
    Rsolver(R_h,err_h,t,y_h);
    free(R_h);
    //printf("y_h: \n");
    //printMatrix(y_h,t,1,t);
    
    cuFloatComplex *y_d, *Q_d;
    CUDA_CALL(cudaMalloc(&y_d,t*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(y_d,y_h,t*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&Q_d,m*t*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(Q_d,Q_h,m*t*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice));
    alpha = make_cuFloatComplex(1,0);
    beta = make_cuFloatComplex(1,0);
    CUBLAS_CALL(cublasCgemv_v2(handle,CUBLAS_OP_N,m,t,&alpha,Q_d,m,y_d,1,&beta,
            x_d,1));
    CUDA_CALL(cudaMemcpy(x_h,x_d,m*sizeof(cuFloatComplex),
            cudaMemcpyDeviceToHost));
    CUBLAS_CALL(cublasDestroy_v2(handle));
    CUDA_CALL(cudaFree(y_d));
    CUDA_CALL(cudaFree(Q_d));
    CUDA_CALL(cudaFree(q_d));
    CUDA_CALL(cudaFree(r_d));
    CUDA_CALL(cudaFree(x_d));
    free(err_h);
    free(h);
    free(s);
    free(c);
    free(H_h);
    free(Q_h);
    return EXIT_SUCCESS;
}


