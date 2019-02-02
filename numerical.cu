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
    cout << inObj(flags,m.numElems) << endl;
    CUDA_CALL(cudaFree(pnts_d));
    CUDA_CALL(cudaFree(elems_d));
    CUDA_CALL(cudaFree(flags_d));
    delete[] flags;
    delete[] dists;
    return EXIT_SUCCESS;
}

ostream& operator<<(ostream &out, const cuFloatComplex &rhs) {
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

ostream& operator<<(ostream &out, const gaussQuad &rhs) {
    out << "Points: " << endl;
    for(int i=0;i<rhs.n;i++) {
        out << rhs.evalPnts[i] << " ";
    }
    out << endl;
    out << "Weights: " << endl;
    for(int i=0;i<rhs.n;i++) {
        out << rhs.wgts[i] << " ";
    }
    return out;
}

