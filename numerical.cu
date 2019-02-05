/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "numerical.h"
#include "mesh.h"

//air density and speed of sound
__constant__ float density = 1.2041;

__constant__ float speed = 343.21;

//Integral points and weights
__constant__ float INTPNTS[INTORDER]; 

__constant__ float INTWGTS[INTORDER];

__global__ void test(cartCoord *pnts, triElem *elems) {
    printf("(%f,%f,%f)\n",pnts[0].coords[0],pnts[0].coords[1],pnts[0].coords[2]);
    printf("%d,%d,%d\n",elems->nodes[0],elems->nodes[1],elems->nodes[2]);
}

int Test() {
    mesh m;
    m.readObj("sphere.obj");
    bool *flags_d;
    CUDA_CALL(cudaMalloc(&flags_d,m.numElems*sizeof(bool)));
    float *dists = new float[m.numPnts];
    bool *flags = new bool[m.numElems];
    cartCoord *pnts_d;
    triElem *elems_d;
    m.meshCloudToGPU(&pnts_d,&elems_d);
    cartCoord sp(0,0,0);
    int width = 32;
    int numBlocks = (m.numElems+width-1)/width;
    cartCoord dirCHIEF(1.1,1,1);
    rayTrnglsInt<<<numBlocks,width>>>(sp,dirCHIEF,pnts_d,elems_d,m.numElems,flags_d);
    CUDA_CALL(cudaMemcpy(flags,flags_d,m.numElems*sizeof(bool),cudaMemcpyDeviceToHost));
    std::cout << inObj(flags,m.numElems) << std::endl;
    CUDA_CALL(cudaFree(pnts_d));
    CUDA_CALL(cudaFree(elems_d));
    CUDA_CALL(cudaFree(flags_d));
    delete[] flags;
    delete[] dists;
    return EXIT_SUCCESS;
}

std::ostream& operator<<(std::ostream &out, const cuFloatComplex &rhs) {
    out << "(" << cuCrealf(rhs) << "," << cuCimagf(rhs) << ")";
    return out;
}

__host__ __device__ cuFloatComplex angExpf(const float theta) {
    return make_cuFloatComplex(cosf(theta),sinf(theta));
}

__host__ __device__ cuFloatComplex expfc(const cuFloatComplex z) {
    cuFloatComplex ans;
    float zr = cuCrealf(z), zi = cuCimagf(z);
    ans = make_cuFloatComplex(exp(zr)*cosf(zi),exp(zr)*sinf(zi));
    return ans;
}

__host__ __device__ cuFloatComplex green(const float k, const float r) {
    float y = 4*PI*r;
    cuFloatComplex x = angExpf(-k*r);
    return make_cuFloatComplex(cuCrealf(x)/y,cuCimagf(x)/y);
}

__host__ __device__ float PsiL(const float radius) {
    return 1.0/(4*PI*radius);
}


__host__ __device__ void printComplexMatrix(cuFloatComplex *A, const int row, const int col, 
        const int lda) {
    float x, y;
    int i, j;
    for (i = 0;i < row;i++) {
        for (j = 0;j < col;j++) {
                x = cuCrealf(A[IDXC0(i,j,lda)]);
                y = cuCimagf(A[IDXC0(i,j,lda)]);
                printf("(%f,%f) ",x,y);
        }
        printf("\n");
    }		
}

__host__ __device__ void printFloatMatrix(float *A, const int row, const int col, const int lda) {
    int i, j;
    for (i = 0;i < row;i++) {
        for (j = 0;j < col;j++) {
            printf("%f ",A[IDXC0(i,j,lda)]);
        }
        printf("\n");
    }	
}

__host__ __device__ float descale(const float lb, const float ub, const float num) {
    return lb+(ub-lb)*num;
}

__host__ __device__ float arrayMin(const float *arr, const int num) {
    float temp = arr[0];
    for(int i=1;i<num;i++) {
        if(arr[i] < temp) {
            temp = arr[i];
        }
    }
    return temp;
}

__host__ __device__ bool inObj(const bool *flags, const int num) {
    int temp = 0;
    for(int i=0;i<num;i++) {
        if(flags[i]) {
            temp++;
        }
    }
    if(temp%2==0) {
        return false;
    } else {
        return true;
    }
}

//Gaussian cartCoords generation
gaussQuad::gaussQuad(const gaussQuad &rhs) {
    n = rhs.n;
    if(evalPnts != NULL) {
        delete[] evalPnts;
    }
    evalPnts = new float[n];
    for(int i=0;i<n;i++) {
        evalPnts[i] = rhs.evalPnts[i];
    }
    if(wgts != NULL) {
        delete[] evalPnts;
    }
    wgts = new float[n];
    for(int i=0;i<n;i++) {
        wgts[i] = rhs.wgts[i];
    }
}

gaussQuad::gaussQuad(const int order): n(order) {
    evalPnts=new float[n];
    wgts=new float[n];
    genGaussParams();
}

gaussQuad::~gaussQuad() {
    if(evalPnts != NULL) {
        delete[] evalPnts;
    }
    if(wgts != NULL) {
        delete[] wgts;
    }
}

gaussQuad& gaussQuad::operator=(const gaussQuad &rhs) {
    n = rhs.n;
    if(evalPnts != NULL) {
        delete[] evalPnts;
    }
    evalPnts = new float[n];
    for(int i=0;i<n;i++) {
        evalPnts[i] = rhs.evalPnts[i];
    }
    if(wgts != NULL) {
        delete[] evalPnts;
    }
    wgts = new float[n];
    for(int i=0;i<n;i++) {
        wgts[i] = rhs.wgts[i];
    }
    return *this;
}

int gaussQuad::genGaussParams() {
    int i, j;
    double t;
    gsl_vector *v = gsl_vector_alloc(n);
    for(i=0;i<n-1;i++) {
        gsl_vector_set(v,i,sqrt(pow(2*(i+1),2)-1));
    }
    for(i=0;i<n-1;i++) {
        t = gsl_vector_get(v,i);
        gsl_vector_set(v,i,(i+1)/t);
    }
    gsl_matrix *A = gsl_matrix_alloc(n,n);
    gsl_matrix *B = gsl_matrix_alloc(n,n);
    for(i=0;i<n;i++) {
        for(j=0;j<n;j++) {
            gsl_matrix_set(A,i,j,0);
            if(i==j) {
                gsl_matrix_set(B,i,j,1);
            } else {
                gsl_matrix_set(B,i,j,0);
            }
        }
    }
    for(i=0;i<n-1;i++) {
        t = gsl_vector_get(v,i);
        gsl_matrix_set(A,i+1,i,t);
        gsl_matrix_set(A,i,i+1,t);
    }
    gsl_eigen_symmv_workspace * wsp = gsl_eigen_symmv_alloc(n);
    HOST_CALL(gsl_eigen_symmv(A,v,B,wsp));
    for(i=0;i<n;i++) {
        evalPnts[i] = gsl_vector_get(v,i);
        t = gsl_matrix_get(B,0,i);
        wgts[i] = 2*pow(t,2);
    }
    gsl_vector_free(v);
    gsl_matrix_free(A);
    gsl_matrix_free(B);
    return EXIT_SUCCESS;
}

int gaussQuad::sendToDevice() {
    CUDA_CALL(cudaMemcpyToSymbol(INTPNTS,evalPnts,INTORDER*sizeof(float),0,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(INTWGTS,wgts,INTORDER*sizeof(float),0,cudaMemcpyHostToDevice));
    return EXIT_SUCCESS;
}

std::ostream& operator<<(std::ostream &out, const gaussQuad &rhs) {
    out << "Points: " << std::endl;
    for(int i=0;i<rhs.n;i++) {
        out << rhs.evalPnts[i] << " ";
    }
    out << std::endl;
    out << "Weights: " << std::endl;
    for(int i=0;i<rhs.n;i++) {
        out << rhs.wgts[i] << " ";
    }
    return out;
}

__host__ int QR_thin(cuFloatComplex *A_h, const int m, const int n, const int lda, 
        const int r, cuFloatComplex *Q_h, const int ldq) {
    //CUDA_CALL(cudaDeviceSynchronize());
    //CUDA_CALL(cudaDeviceReset());
    if(m<=n) {
        printf("Not a thin matrix, Error.\n");
        return EXIT_FAILURE;
    }
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));
    CUBLAS_CALL(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST)); //switch to host pointer mode
    int i, j, k, u, s;
    cuFloatComplex *A_d, *Q_d; //global memory for A_h and Q_h memory space
    CUDA_CALL(cudaMalloc((void**)&A_d,m*n*sizeof(cuFloatComplex)));
    CUBLAS_CALL(cublasSetMatrix(m,n,sizeof(cuFloatComplex),A_h,lda,A_d,m));
    CUDA_CALL(cudaMalloc((void**)&Q_d,m*m*sizeof(cuFloatComplex)));
    for(i=0;i<m;i++) {
        for(j=0;j<m;j++) {
            if(i == j) {
                Q_h[IDXC0(i,j,ldq)] = make_cuFloatComplex(1,0);
            }
            else {
                Q_h[IDXC0(i,j,ldq)] = make_cuFloatComplex(0,0);
            }
        }
    }
    CUBLAS_CALL(cublasSetMatrix(m,m,sizeof(cuFloatComplex),Q_h,ldq,Q_d,m));
    cublasGetMatrix(m,m,sizeof(cuFloatComplex),Q_d,m,Q_h,ldq);
    //printf("The Initialized Q_d is: \n");
    //printMatrix_complex(Q_h,m,m,ldq);
    //printf("\n");
    
    cuFloatComplex *v_d;
    CUDA_CALL(cudaMalloc((void**)&v_d,m*sizeof(cuFloatComplex)));
    cuFloatComplex *v_h;
    v_h = (cuFloatComplex*)malloc(m*sizeof(cuFloatComplex));
    float beta;
    cuFloatComplex x1, temp, temp_alpha, temp_beta;
    float x1_r, x1_i, x1_mag, x_nrm;
    
    cuFloatComplex *prod_bvA; //for saving the intermediate result for beta*v^H*A()
    CUDA_CALL(cudaMalloc((void**)&prod_bvA,r*sizeof(cuFloatComplex)));
    cuFloatComplex *prod_bvA_h;
    prod_bvA_h = (cuFloatComplex*)malloc(r*sizeof(cuFloatComplex));
    
    cuFloatComplex *V_d;
    CUDA_CALL(cudaMalloc((void**)&V_d,m*r*sizeof(cuFloatComplex)));
    float *B_h;
    B_h = (float*)malloc(r*sizeof(float));
    cuFloatComplex *zeros_h;
    zeros_h = (cuFloatComplex*)malloc(r*sizeof(cuFloatComplex));
    for(i=0;i<r;i++) {
        zeros_h[i] = make_cuFloatComplex(0,0);
    }
    
    cuFloatComplex *Y_d, *W_d;
    CUDA_CALL(cudaMalloc((void**)&Y_d,m*r*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMalloc((void**)&W_d,m*r*sizeof(cuFloatComplex)));
    cuFloatComplex *prod_Yv; //for saving the product of Y and v, of length r;
    CUDA_CALL(cudaMalloc((void**)&prod_Yv,r*sizeof(cuFloatComplex)));
    cuFloatComplex *prod_Yv_h;
    prod_Yv_h = (cuFloatComplex*)malloc(r*sizeof(cuFloatComplex));
    cuFloatComplex *z_d; //aid vector
    CUDA_CALL(cudaMalloc((void**)&z_d,m*sizeof(cuFloatComplex)));
    cuFloatComplex *prod_WA, *prod_QW; //intermediate results
    CUDA_CALL(cudaMalloc((void**)&prod_WA,r*n*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMalloc((void**)&prod_QW,m*r*sizeof(cuFloatComplex)));
    
    for(k=1;k<=n/r;k++) {
        //printf("k=%d\n",k);
        s = (k-1)*r+1; //the first colum in the current block
        for(j=1;j<=r;j++) {
            u = s+j-1;
            //printf("u=%d\n",u);
            CUDA_CALL(cudaMemcpy(&x1,&A_d[IDXC1(u,u,m)],sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
            //printf("x1=(%f,%f)\n",cuCrealf(x1),cuCimagf(x1));
            x1_r = cuCrealf(x1);
            x1_i = cuCimagf(x1);
            x1_mag = sqrtf(powf(x1_r,2)+powf(x1_i,2));
            x1_r = x1_r/x1_mag;
            x1_i = x1_i/x1_mag;
            CUBLAS_CALL(cublasScnrm2(handle,m-u+1,&A_d[IDXC1(u,u,m)],1,&x_nrm));
            //printf("x_nrm = %f\n", x_nrm);
            temp = make_cuFloatComplex(x1_r*x_nrm,x1_i*x_nrm);
            temp = cuCsubf(x1,temp);
            CUDA_CALL(cudaMemcpy(v_d,&A_d[IDXC1(u,u,m)],(m-u+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
            CUDA_CALL(cudaMemcpy(v_d,&temp,sizeof(cuFloatComplex),cudaMemcpyHostToDevice)); //v in house transform derived
            //CUDA_CALL(cudaMemcpy(v_h,v_d,(m-u+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
            //printf("v_d: \n");
            //printMatrix_complex(v_h,1,m-u+1,1);
            CUBLAS_CALL(cublasScnrm2(handle,m-u+1,v_d,1,&beta));
            beta = 2.0/powf(beta,2);
            //printf("beta = %f\n",beta);
            temp_alpha = make_cuFloatComplex(beta,0);
            temp_beta = make_cuFloatComplex(0,0);
            CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_C,CUBLAS_OP_N,1,s+r-u,m-u+1,&temp_alpha,v_d,m,&A_d[IDXC1(u,u,m)],m,
                    &temp_beta,prod_bvA,1));
            //CUDA_CALL(cudaMemcpy(prod_bvA_h,prod_bvA,(s+r-u)*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
            //printf("prod_bvA: \n");
            //printMatrix_complex(prod_bvA_h,1,s+r-u,1);
            //printf("\n");
            temp_alpha = make_cuFloatComplex(-1,0);
            temp_beta = make_cuFloatComplex(1,0);
            CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m-u+1,s+r-u,1,&temp_alpha,v_d,m,prod_bvA,1,
                    &temp_beta,&A_d[IDXC1(u,u,m)],m));
            CUDA_CALL(cudaMemcpy(&V_d[IDXC1(j,j,m)],v_d,(m-u+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
            CUBLAS_CALL(cublasSetVector(j-1,sizeof(cuFloatComplex),zeros_h,1,&V_d[IDXC1(1,j,m)],1));
            B_h[j-1] = beta;
            
        }
        CUDA_CALL(cudaMemcpy(Y_d,V_d,(m-s+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
        temp_alpha = make_cuFloatComplex(-B_h[0],0);
        CUBLAS_CALL(cublasSetVector(m-s+1,sizeof(cuFloatComplex),V_d,1,W_d,1));
        CUBLAS_CALL(cublasCscal(handle,m-s+1,&temp_alpha,W_d,1));
        for(j=2;j<=r;j++) {
            CUDA_CALL(cudaMemcpy(v_d,&V_d[IDXC1(1,j,m)],(m-s+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
            temp_alpha = make_cuFloatComplex(1,0);
            temp_beta = make_cuFloatComplex(0,0);
            CUBLAS_CALL(cublasCgemv(handle,CUBLAS_OP_C,m-s+1,j-1,&temp_alpha,Y_d,m,v_d,1,&temp_beta,prod_Yv,1));
            //CUDA_CALL(cudaMemcpy(prod_Yv_h,prod_Yv,(j-1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
            //printf("Yv = \n");
            //printMatrix_complex(prod_Yv_h,1,j-1,1);
            temp_alpha = make_cuFloatComplex(-B_h[j-1],0);
            temp_beta = make_cuFloatComplex(-B_h[j-1],0);
            CUDA_CALL(cudaMemcpy(z_d,v_d,(m-s+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
            CUBLAS_CALL(cublasCgemv(handle,CUBLAS_OP_N,m-s+1,j-1,&temp_alpha,W_d,m,prod_Yv,1,&temp_beta,z_d,1));
            CUDA_CALL(cudaMemcpy(&W_d[IDXC1(1,j,m)],z_d,(m-s+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
            CUDA_CALL(cudaMemcpy(&Y_d[IDXC1(1,j,m)],v_d,(m-s+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
        }
        if(s+r <= n) {
            temp_alpha = make_cuFloatComplex(1,0);
            temp_beta = make_cuFloatComplex(0,0);
            CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_C,CUBLAS_OP_N,r,n-s-r+1,m-s+1,&temp_alpha,W_d,m,&A_d[IDXC1(s,s+r,m)],m,
                    &temp_beta,prod_WA,r));
            temp_alpha = make_cuFloatComplex(1,0);
            temp_beta = make_cuFloatComplex(1,0);
            CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m-s+1,n-s-r+1,r,&temp_alpha,Y_d,m,prod_WA,r,
                    &temp_beta,&A_d[IDXC1(s,s+r,m)],m));
            //CUBLAS_CALL(cublasGetMatrix(m,n,sizeof(cuFloatComplex),A_d,m,A_h,lda));
            //printf("The current A is: \n");
            //printMatrix_complex(A_h,m,n,lda);
            //printf("\n");
        }
        temp_alpha = make_cuFloatComplex(1,0);
        temp_beta = make_cuFloatComplex(0,0);
        CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,r,m-s+1,&temp_alpha,&Q_d[IDXC1(1,s,m)],m,W_d,m,
                &temp_beta,prod_QW,m));
        temp_alpha = make_cuFloatComplex(1,0);
        temp_beta = make_cuFloatComplex(1,0);
        CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_N,CUBLAS_OP_C,m,m-s+1,r,&temp_alpha,prod_QW,m,Y_d,m,
                &temp_beta,&Q_d[IDXC1(1,s,m)],m));
    }
    //printf("k=%d, k*r=%d\n",k,k*r);
    if(r*(n/r) < n) { // there still remains columns unprocessed
        //printf("Entering the remaining block.\n");
        s = r*(n/r)+1;
        for(j=1;j<=n-r*(n/r);j++) {
            u = s+j-1;
            //printf("u=%d\n",u);
            CUDA_CALL(cudaMemcpy(&x1,&A_d[IDXC1(u,u,m)],sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
            //printf("x1=(%f,%f)\n",cuCrealf(x1),cuCimagf(x1));
            x1_r = cuCrealf(x1);
            x1_i = cuCimagf(x1);
            x1_mag = sqrtf(powf(x1_r,2)+powf(x1_i,2));
            x1_r = x1_r/x1_mag;
            x1_i = x1_i/x1_mag;
            CUBLAS_CALL(cublasScnrm2(handle,m-u+1,&A_d[IDXC1(u,u,m)],1,&x_nrm));
            //printf("x_nrm = %f\n", x_nrm);
            temp = make_cuFloatComplex(x1_r*x_nrm,x1_i*x_nrm);
            temp = cuCsubf(x1,temp);
            CUDA_CALL(cudaMemcpy(v_d,&A_d[IDXC1(u,u,m)],(m-u+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
            CUDA_CALL(cudaMemcpy(v_d,&temp,sizeof(cuFloatComplex),cudaMemcpyHostToDevice)); //v in house transform derived
            //CUDA_CALL(cudaMemcpy(v_h,v_d,(m-u+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
            //printf("v_d: \n");
            //printMatrix_complex(v_h,1,m-u+1,1);
            CUBLAS_CALL(cublasScnrm2(handle,m-u+1,v_d,1,&beta));
            beta = 2.0/powf(beta,2);
            //printf("beta = %f\n",beta);
            temp_alpha = make_cuFloatComplex(beta,0);
            temp_beta = make_cuFloatComplex(0,0);
            CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_C,CUBLAS_OP_N,1,s+(n-r*(n/r))-u,m-u+1,&temp_alpha,v_d,m,&A_d[IDXC1(u,u,m)],m,
                    &temp_beta,prod_bvA,1));
            //CUDA_CALL(cudaMemcpy(prod_bvA_h,prod_bvA,(s+r-u)*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
            //printf("prod_bvA: \n");
            //printMatrix_complex(prod_bvA_h,1,s+(n-r*(n/r))-u,1);
            //printf("\n");
            temp_alpha = make_cuFloatComplex(-1,0);
            temp_beta = make_cuFloatComplex(1,0);
            CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m-u+1,s+(n-r*(n/r))-u,1,&temp_alpha,v_d,m,prod_bvA,1,
                    &temp_beta,&A_d[IDXC1(u,u,m)],m));
            CUDA_CALL(cudaMemcpy(&V_d[IDXC1(j,j,m)],v_d,(m-u+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
            CUBLAS_CALL(cublasSetVector(j-1,sizeof(cuFloatComplex),zeros_h,1,&V_d[IDXC1(1,j,m)],1));
            B_h[j-1] = beta;
            
        }
        CUDA_CALL(cudaMemcpy(Y_d,V_d,(m-s+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
        temp_alpha = make_cuFloatComplex(-B_h[0],0);
        CUBLAS_CALL(cublasSetVector(m-s+1,sizeof(cuFloatComplex),V_d,1,W_d,1));
        CUBLAS_CALL(cublasCscal(handle,m-s+1,&temp_alpha,W_d,1));
        for(j=2;j<=n-r*(n/r);j++) {
            CUDA_CALL(cudaMemcpy(v_d,&V_d[IDXC1(1,j,m)],(m-s+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
            temp_alpha = make_cuFloatComplex(1,0);
            temp_beta = make_cuFloatComplex(0,0);
            CUBLAS_CALL(cublasCgemv(handle,CUBLAS_OP_C,m-s+1,j-1,&temp_alpha,Y_d,m,v_d,1,&temp_beta,prod_Yv,1));
            //CUDA_CALL(cudaMemcpy(prod_Yv_h,prod_Yv,(j-1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
            //printf("Yv = \n");
            //printMatrix_complex(prod_Yv_h,1,j-1,1);
            temp_alpha = make_cuFloatComplex(-B_h[j-1],0);
            temp_beta = make_cuFloatComplex(-B_h[j-1],0);
            CUDA_CALL(cudaMemcpy(z_d,v_d,(m-s+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
            CUBLAS_CALL(cublasCgemv(handle,CUBLAS_OP_N,m-s+1,j-1,&temp_alpha,W_d,m,prod_Yv,1,&temp_beta,z_d,1));
            CUDA_CALL(cudaMemcpy(&W_d[IDXC1(1,j,m)],z_d,(m-s+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
            CUDA_CALL(cudaMemcpy(&Y_d[IDXC1(1,j,m)],v_d,(m-s+1)*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
        }
        if(s+r <= n) {
            temp_alpha = make_cuFloatComplex(1,0);
            temp_beta = make_cuFloatComplex(0,0);
            CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_C,CUBLAS_OP_N,r,n-s-r+1,m-s+1,&temp_alpha,W_d,m,&A_d[IDXC1(s,s+r,m)],m,
                    &temp_beta,prod_WA,r));
            temp_alpha = make_cuFloatComplex(1,0);
            temp_beta = make_cuFloatComplex(1,0);
            CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m-s+1,n-s-r+1,r,&temp_alpha,Y_d,m,prod_WA,r,
                    &temp_beta,&A_d[IDXC1(s,s+r,m)],m));
            //CUBLAS_CALL(cublasGetMatrix(m,n,sizeof(cuFloatComplex),A_d,m,A_h,lda));
            //printf("The current A is: \n");
            //printMatrix_complex(A_h,m,n,lda);
            //printf("\n");
        }
        temp_alpha = make_cuFloatComplex(1,0);
        temp_beta = make_cuFloatComplex(0,0);
        CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n-r*(n/r),m-s+1,&temp_alpha,&Q_d[IDXC1(1,s,m)],m,W_d,m,
                &temp_beta,prod_QW,m));
        temp_alpha = make_cuFloatComplex(1,0);
        temp_beta = make_cuFloatComplex(1,0);
        CUBLAS_CALL(cublasCgemm(handle,CUBLAS_OP_N,CUBLAS_OP_C,m,m-s+1,n-r*(n/r),&temp_alpha,prod_QW,m,Y_d,m,
                &temp_beta,&Q_d[IDXC1(1,s,m)],m));
    }
    CUBLAS_CALL(cublasGetMatrix(m,n,sizeof(cuFloatComplex),A_d,m,A_h,lda));
    CUBLAS_CALL(cublasGetMatrix(m,m,sizeof(cuFloatComplex),Q_d,m,Q_h,ldq));
    free(B_h);
    free(zeros_h);
    free(prod_Yv_h);
    free(v_h);
    free(prod_bvA_h);
    CUDA_CALL(cudaFree(A_d));
    CUDA_CALL(cudaFree(Q_d));
    CUDA_CALL(cudaFree(v_d));
    CUDA_CALL(cudaFree(V_d));
    CUDA_CALL(cudaFree(z_d));
    CUDA_CALL(cudaFree(Y_d));
    CUDA_CALL(cudaFree(W_d));
    CUDA_CALL(cudaFree(prod_QW));
    CUDA_CALL(cudaFree(prod_WA));
    CUDA_CALL(cudaFree(prod_Yv));
    CUDA_CALL(cudaFree(prod_bvA));
    CUBLAS_CALL(cublasDestroy(handle));
    //CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}

__host__ int lsqSolver(cuFloatComplex *A_h, const int m, const int n, const int lda,
        cuFloatComplex *B_h, const int nrhs, const int ldb, cuFloatComplex *Q_h) {
    if(m>n) {
        cuFloatComplex *temp;
        //printf("A_h: \n");
        //printMatrix_complex(A_h,10,10,lda);
        //printf("Q_h: \n");
        //printMatrix_complex(Q_h,10,10,m);
        HOST_CALL(QR_thin(A_h,m,n,lda,256,Q_h,m));
	CUDA_CALL(cudaDeviceSynchronize());
        //printf("A_h: \n");
        //printMatrix_complex(A_h,10,10,lda);
        //printf("Q_h: \n");
        //printMatrix_complex(Q_h,10,10,m);
        //CUDA_CALL(cudaDeviceReset());
        cublasHandle_t cublasH;
        cublasCreate(&cublasH);
        cublasSetPointerMode(cublasH,CUBLAS_POINTER_MODE_HOST);
        cuFloatComplex *Q_d, *R_d, *B_d;
        CUDA_CALL(cudaMalloc((void**)&Q_d,m*m*sizeof(cuFloatComplex)));
        CUDA_CALL(cudaMalloc((void**)&R_d,m*n*sizeof(cuFloatComplex)));
        CUDA_CALL(cudaMalloc((void**)&B_d,m*nrhs*sizeof(cuFloatComplex)));
        CUBLAS_CALL(cublasSetMatrix(m,n,sizeof(cuFloatComplex),A_h,lda,R_d,m));
        CUBLAS_CALL(cublasSetMatrix(m,m,sizeof(cuFloatComplex),Q_h,m,Q_d,m));
        CUBLAS_CALL(cublasSetMatrix(m,nrhs,sizeof(cuFloatComplex),B_h,ldb,B_d,m));
        CUDA_CALL(cudaMalloc((void**)&temp,m*nrhs*sizeof(cuFloatComplex)));
        CUDA_CALL(cudaMemcpy(temp,B_d,m*nrhs*sizeof(cuFloatComplex),cudaMemcpyDeviceToDevice));
        cuFloatComplex alpha = make_cuFloatComplex(1,0), beta = make_cuFloatComplex(0,0);
        CUBLAS_CALL(cublasCgemm(cublasH,CUBLAS_OP_C,CUBLAS_OP_N,m,nrhs,m,&alpha,Q_d,m,temp,m,
                &beta,B_d,m));
        CUDA_CALL(cudaFree(temp));
        CUBLAS_CALL(cublasCtrsm(cublasH,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,n,nrhs,&alpha,R_d,m,B_d,m));
        CUBLAS_CALL(cublasGetMatrix(n,nrhs,sizeof(cuFloatComplex),B_d,m,B_h,ldb));
        CUDA_CALL(cudaFree(Q_d));
        CUDA_CALL(cudaFree(R_d));
        CUDA_CALL(cudaFree(B_d));
        
        CUBLAS_CALL(cublasDestroy(cublasH));
        
    }
    else {
        printf("Senario not included yet.");
        return EXIT_FAILURE;
    }
    CUDA_CALL(cudaDeviceSynchronize());
    //CUDA_CALL(cudaDeviceReset());
    return EXIT_SUCCESS;
}



