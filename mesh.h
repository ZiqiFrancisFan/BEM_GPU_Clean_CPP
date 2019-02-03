/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   mesh.h
 * Author: ziqi
 *
 * Created on January 29, 2019, 10:26 PM
 */

#ifndef MESH_H
#define MESH_H
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>



using namespace std;

class cartCoord2D;
class triElem;
class mesh;

//class cartCoord
class cartCoord {
    friend ostream& operator<<(ostream&,const cartCoord&);
    
    friend __host__ __device__ cartCoord numDvd(const cartCoord&,const float);
    
    friend __host__ __device__ cartCoord numMul(const float,const cartCoord&);
    
    friend __host__ __device__ float dotProd(const cartCoord&,const cartCoord&);
    
    friend __host__ __device__ cuFloatComplex green2(const float,const cartCoord,const cartCoord);
    
    friend __host__ __device__ float Psi_L(const cartCoord);
    
    friend __host__ __device__ cartCoord xiToElem(const cartCoord,const cartCoord,
        const cartCoord,const cartCoord2D);
    
    friend __host__ __device__ float trnglArea(const cartCoord,const cartCoord);
    
    friend __host__ __device__ cartCoord rayPlaneInt(const cartCoord,const cartCoord,
    const cartCoord,const cartCoord);
    
    friend __host__ __device__ bool rayTrnglInt(const cartCoord,const cartCoord,
    const cartCoord,const cartCoord,const cartCoord);
    
    friend __host__ __device__ float r(const cartCoord,const cartCoord);
    
    friend __host__ __device__ float prpn2(const cartCoord,const cartCoord,const cartCoord);
    
    friend __global__ void rayTrnglsInt(const cartCoord,const cartCoord,
    const cartCoord*,const triElem*,const int,bool*);
    
    friend __global__ void test(cartCoord *pnts, triElem *elems);
    
    friend __global__ void distPntPnts(const cartCoord,const cartCoord*,const int,float*);
    
    friend __host__ __device__ float prRpn2(const cartCoord,const cartCoord,const cartCoord);
    
    friend __host__ __device__ cuFloatComplex pGpn2(const float,const cartCoord,const cartCoord,const cartCoord);
    
    friend __host__ __device__ float PsiL2(const cartCoord,const cartCoord);
    
    friend __host__ __device__ float pPsiLpn2(const cartCoord,const cartCoord,const cartCoord);
    
    friend __global__ void pntsElem_lnm_nsgl(const float k, const int l, const int n, const int m, 
        const triElem *elems, const cartCoord *pnts, const int numPnts, 
        cuFloatComplex *hCoeffs, cuFloatComplex *gCoeffs, float *cCoeffs);
    
    friend __global__ void pntsElems_nm_sgl(const float k, const int n, const int m, const triElem *elems, 
        const int numElems, const cartCoord *pnts, cuFloatComplex *hCoeffs_sgl1, 
        cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, cuFloatComplex *gCoeffs_sgl1, 
        cuFloatComplex *gCoeffs_sgl2, cuFloatComplex *gCoeffs_sgl3, float *cCoeffs_sgl1, 
        float *cCoeffs_sgl2, float *cCoeffs_sgl3);
    
    friend class mesh;
private:
    float coords[3];

public:
    __host__ __device__ cartCoord() {coords[0]=0;coords[1]=0;coords[2]=0;}
    __host__ __device__ cartCoord(const cartCoord&);
    __host__ __device__ cartCoord(const float x,const float y,const float z) {coords[0]=x;coords[1]=y;coords[2]=z;}
    __host__ __device__ cartCoord& operator=(const cartCoord&);
    ~cartCoord() = default;
    __host__ __device__ void set(const float,const float,const float);
    __host__ __device__ cartCoord operator+(const cartCoord&) const;
    __host__ __device__ cartCoord operator-(const cartCoord&) const;
    __host__ __device__ cartCoord operator-() const;
    __host__ __device__ cartCoord operator*(const cartCoord&) const;
    __host__ __device__ void print() {printf("(%f,%f,%f)\n",coords[0],coords[1],coords[2]);}
    __host__ __device__ float nrm2() const;
    __host__ __device__ cartCoord nrmlzd();
    __host__ __device__ bool isEqual(const cartCoord) const;
    __host__ __device__ bool isLegal() const;
    __host__ __device__ bool isInsideTrngl(const cartCoord,const cartCoord,const cartCoord) const; 
    
};

ostream& operator<<(ostream&,const cartCoord&);

__host__ __device__ cartCoord numDvd(const cartCoord&,const float);

__host__ __device__ cartCoord numMul(const float,const cartCoord&);

__host__ __device__ float dotProd(const cartCoord&,const cartCoord&);

__host__ __device__ cuFloatComplex green(const cartCoord&,const cartCoord&);

__host__ __device__ float Psi_L(const cartCoord);

__host__ __device__ float trnglArea(const cartCoord,const cartCoord);

__host__ __device__ cartCoord rayPlaneInt(const cartCoord,const cartCoord,
    const cartCoord,const cartCoord);

__host__ __device__ bool rayTrnglInt(const cartCoord,const cartCoord,
    const cartCoord,const cartCoord,const cartCoord);

__host__ __device__ float r(const cartCoord,const cartCoord);

__host__ __device__ float prpn2(const cartCoord,const cartCoord,const cartCoord);

__host__ __device__ float prRpn2(const cartCoord,const cartCoord,const cartCoord);

__host__ __device__ cuFloatComplex pGpn2(const float,const cartCoord,const cartCoord,const cartCoord);

__host__ __device__ float PsiL2(const cartCoord,const cartCoord);

__host__ __device__ float pPsiLpn2(const cartCoord,const cartCoord,const cartCoord);

//class triElem
class triElem {
    friend ostream& operator<<(ostream&,const triElem&);
    
    friend __global__ void test(cartCoord *pnts, triElem *elems);
    
    friend __global__ void rayTrnglsInt(const cartCoord,const cartCoord,
    const cartCoord*,const triElem*,const int,bool*);
    
    friend __global__ void pntsElem_lnm_nsgl(const float k, const int l, const int n, const int m, 
        const triElem *elems, const cartCoord *pnts, const int numPnts, 
        cuFloatComplex *hCoeffs, cuFloatComplex *gCoeffs, float *cCoeffs);
    
    friend __global__ void pntsElems_nm_sgl(const float k, const int n, const int m, const triElem *elems, 
        const int numElems, const cartCoord *pnts, cuFloatComplex *hCoeffs_sgl1, 
        cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, cuFloatComplex *gCoeffs_sgl1, 
        cuFloatComplex *gCoeffs_sgl2, cuFloatComplex *gCoeffs_sgl3, float *cCoeffs_sgl1, 
        float *cCoeffs_sgl2, float *cCoeffs_sgl3);
    
    friend __global__ void updateSystemLhs_hg_nsgl(cuFloatComplex *A, const int numPnts, const int numCHIEFs, 
        const int lda, const cuFloatComplex *hCoeffs, const cuFloatComplex *gCoeffs, 
        const triElem *elems, const int l);
    
    friend __global__ void updateSystemLhs_c_nsgl(cuFloatComplex *A, const int numPnts, const int lda, 
        const float *cCoeffs);
    
    friend __global__ void updateSystemRhs_nsgl(cuFloatComplex *B, const int numPnts, const int numCHIEF, 
        const int ldb, const int srcIdx, const cuFloatComplex *gCoeffs, const triElem *elems, 
        const int l);
    
    friend __global__ void updateSystemLhs_hg_sgl(cuFloatComplex *A, 
        const int lda, cuFloatComplex *hCoeffs_sgl1, cuFloatComplex *hCoeffs_sgl2, 
        cuFloatComplex *hCoeffs_sgl3, cuFloatComplex *gCoeffs_sgl1, cuFloatComplex *gCoeffs_sgl2, 
        cuFloatComplex *gCoeffs_sgl3, const triElem *elems, const int numElems);
    
    friend __global__ void updateSystemLhs_c_sgl(cuFloatComplex *A, const int lda, float *cCoeffs_sgl1, 
        float *cCoeffs_sgl2, float *cCoeffs_sgl3, const triElem *elems, const int numElems);
    
    friend class mesh;
private:
    int nodes[3];
    cuFloatComplex bc[3];
    
public:
    __host__ __device__ triElem() {}
    __host__ __device__ triElem(const triElem&);
    __host__ __device__ ~triElem() {}
    __host__ __device__ triElem& operator=(const triElem&);
};

ostream& operator<<(ostream&,const triElem&);



//class mesh
class mesh {
    friend ostream& operator<<(ostream&,const mesh&);
    friend __global__ void rayTrnglsInt(const cartCoord,const cartCoord,
    const cartCoord*,const triElem*,const int,bool*);
    friend int Test();
    
private:
    cartCoord *pnts = NULL;
    cartCoord *chiefPnts = NULL;
    triElem *elems = NULL;
    int numPnts = 0;
    int numCHIEF = 0;
    int numElems = 0;
    
    cartCoord dirCHIEF;
    
    float xl=0, xu=0, yl=0, yu=0, zl=0, zu=0;
    
public:
    mesh() = default;
    mesh(const mesh&);
    ~mesh();
    int readObj(const char*);
    mesh& operator=(const mesh&);
    int findBB(const float);
    int meshCloudToGPU(cartCoord**,triElem**);
    int genCHIEF(const int,const float);
    void printBB();
    int chiefToGPU(cartCoord**);
    int meshToGPU(cartCoord**,triElem**);
};

ostream& operator<<(ostream&,const mesh&);

__global__ void rayTrnglsInt(const cartCoord*,const triElem*,bool*);

class cartCoord2D {
    friend __host__ __device__ cartCoord2D numDvd(const cartCoord2D&,const float);
    friend __host__ __device__ cartCoord2D numMul(const float,const cartCoord2D&);
    friend __host__ __device__ float N_1(const cartCoord2D);
    friend __host__ __device__ float N_2(const cartCoord2D);
    friend __host__ __device__ float N_3(const cartCoord2D);
    friend __host__ __device__ float pN1pXi1(const cartCoord2D);
    friend __host__ __device__ float pN1pXi2(const cartCoord2D);
    friend __host__ __device__ float pN2pXi1(const cartCoord2D);
    friend __host__ __device__ float pN2pXi2(const cartCoord2D);
    friend __host__ __device__ float pN3pXi1(const cartCoord2D);
    friend __host__ __device__ float pN3pXi2(const cartCoord2D);
    friend __host__ __device__ cartCoord2D etaToRhoTheta(const cartCoord2D);
    friend __host__ __device__ cartCoord2D rhoThetaToXi_3(const cartCoord2D);
    friend __host__ __device__ cartCoord2D rhoThetaToXi_1(const cartCoord2D);
    friend __host__ __device__ cartCoord2D rhoThetaToXi_2(const cartCoord2D);
    friend __host__ __device__ cartCoord xiToElem(const cartCoord,const cartCoord,
        const cartCoord,const cartCoord2D);
    
private:
    float coords[2];
    
public:
    __host__ __device__ cartCoord2D() {coords[0]=0;coords[1]=0;}
    __host__ __device__ cartCoord2D(const cartCoord2D&);
    __host__ __device__ cartCoord2D(const float x,const float y) {coords[0]=x;coords[1]=y;}
    ~cartCoord2D() = default;
    __host__ __device__ cartCoord2D& operator=(const cartCoord2D&);
    __host__ __device__ cartCoord2D etaToRhoTheta(const cartCoord2D);
    __host__ __device__ void set(const float,const float);
    __host__ __device__ cartCoord2D operator+(const cartCoord2D&) const;
    __host__ __device__ cartCoord2D operator-(const cartCoord2D&) const;
    __host__ __device__ void print() {printf("(%f,%f)\n",coords[0],coords[1]);}
};

__host__ __device__ cartCoord2D numDvd(const cartCoord2D&,const float);

__host__ __device__ cartCoord2D numMul(const float,const cartCoord2D&);

__host__ __device__ float N_1(const cartCoord2D);

__host__ __device__ float N_2(const cartCoord2D);

__host__ __device__ float N_3(const cartCoord2D);

__host__ __device__ float pN1pXi1(const cartCoord2D);

__host__ __device__ float pN1pXi2(const cartCoord2D);

__host__ __device__ float pN2pXi1(const cartCoord2D);

__host__ __device__ float pN2pXi2(const cartCoord2D);

__host__ __device__ float pN3pXi1(const cartCoord2D);

__host__ __device__ float pN3pXi2(const cartCoord2D);

__host__ __device__ cartCoord xiToElem(const cartCoord,const cartCoord,
        const cartCoord,const cartCoord2D);

__host__ __device__ cartCoord2D etaToRhoTheta(const cartCoord2D);

__host__ __device__ cartCoord2D rhoThetaToXi_3(const cartCoord2D);

__host__ __device__ cartCoord2D rhoThetaToXi_1(const cartCoord2D);

__host__ __device__ cartCoord2D rhoThetaToXi_2(const cartCoord2D);

__device__ cuFloatComplex g_lnm_1_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex g_lnm_2_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex g_lnm_3_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_1_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_2_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_3_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex g_lnm_1_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex g_lnm_2_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex g_lnm_3_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex g_lnm_1_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex g_lnm_2_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex g_lnm_3_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex g_lnm_1_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex g_lnm_2_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex g_lnm_3_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_1_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_2_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_3_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_1_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_2_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_3_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_1_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_2_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ cuFloatComplex h_lnm_3_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m);

__device__ float c_l_nsgl(const cartCoord x, const cartCoord p1, const cartCoord p2, 
        const cartCoord p3, const int n, const int m);

__global__ void pntsElem_lnm_nsgl(const float k, const int l, const int n, const int m, 
        const triElem *elems, const cartCoord *pnts, const int numPnts, 
        cuFloatComplex *hCoeffs, cuFloatComplex *gCoeffs, float *cCoeffs);

__global__ void pntsElems_nm_sgl(const float k, const int n, const int m, const triElem *elems, 
        const int numElems, const cartCoord *pnts, cuFloatComplex *hCoeffs_sgl1, 
        cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, cuFloatComplex *gCoeffs_sgl1, 
        cuFloatComplex *gCoeffs_sgl2, cuFloatComplex *gCoeffs_sgl3, float *cCoeffs_sgl1, 
        float *cCoeffs_sgl2, float *cCoeffs_sgl3);

__global__ void updateSystemLhs_hg_nsgl(cuFloatComplex *A, const int numPnts, const int numCHIEFs, 
        const int lda, const cuFloatComplex *hCoeffs, const cuFloatComplex *gCoeffs, 
        const triElem *elems, const int l);

__global__ void updateSystemLhs_c_nsgl(cuFloatComplex *A, const int numPnts, const int lda, 
        const float *cCoeffs);

__global__ void updateSystemRhs_nsgl(cuFloatComplex *B, const int numPnts, const int numCHIEF, 
        const int ldb, const int srcIdx, const cuFloatComplex *gCoeffs, const triElem *elems, 
        const int l);

__global__ void updateSystemLhs_hg_sgl(cuFloatComplex *A, 
        const int lda, cuFloatComplex *hCoeffs_sgl1, cuFloatComplex *hCoeffs_sgl2, 
        cuFloatComplex *hCoeffs_sgl3, cuFloatComplex *gCoeffs_sgl1, cuFloatComplex *gCoeffs_sgl2, 
        cuFloatComplex *gCoeffs_sgl3, const triElem *elems, const int numElems);

__global__ void updateSystemLhs_c_sgl(cuFloatComplex *A, const int lda, float *cCoeffs_sgl1, 
        float *cCoeffs_sgl2, float *cCoeffs_sgl3, const triElem *elems, const int numElems);

__device__ cuFloatComplex g_l_1_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex g_l_2_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex g_l_3_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_1_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_2_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_3_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex g_l_1_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex g_l_2_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex g_l_3_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex g_l_1_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex g_l_2_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex g_l_3_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex g_l_1_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex g_l_2_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex g_l_3_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_1_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_2_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_3_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_1_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_2_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_3_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_1_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_2_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);

__device__ cuFloatComplex h_l_3_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3);
#endif /* MESH_H */

