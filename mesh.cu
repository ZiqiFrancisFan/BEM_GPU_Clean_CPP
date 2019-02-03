/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <cuComplex.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <unistd.h>
#include<ctime>
#include "mesh.h"
#include "numerical.h"

//cartCoord class functions
__host__ __device__ cartCoord::cartCoord(const cartCoord &rhs) {
    for(int i=0;i<3;i++) {
        coords[i] = rhs.coords[i];
    }
}

__host__ __device__ void cartCoord::set(const float x, const float y, const float z) {
    coords[0] = x;
    coords[1] = y;
    coords[2] = z;
}

ostream& operator<<(ostream &out,const cartCoord &rhs) {
    out << "(" << rhs.coords[0] << "," << rhs.coords[1] << "," << rhs.coords[2] 
            << ")";
    return out;
}

__host__ __device__ cartCoord& cartCoord::operator=(const cartCoord &rhs) {
    coords[0] = rhs.coords[0];
    coords[1] = rhs.coords[1];
    coords[2] = rhs.coords[2];
    return *this;
}

__host__ __device__ cartCoord cartCoord::operator+(const cartCoord &rhs) const {
    cartCoord temp;
    temp.coords[0] = coords[0] + rhs.coords[0];
    temp.coords[1] = coords[1] + rhs.coords[1];
    temp.coords[2] = coords[2] + rhs.coords[2];
    return temp;
}

__host__ __device__ cartCoord cartCoord::operator-(const cartCoord &rhs) const {
    cartCoord temp;
    temp.coords[0] = coords[0] - rhs.coords[0];
    temp.coords[1] = coords[1] - rhs.coords[1];
    temp.coords[2] = coords[2] - rhs.coords[2];
    return temp;
}

__host__ __device__ cartCoord cartCoord::operator-() const {
    return cartCoord(-coords[0],-coords[1],-coords[2]);
}

__host__ __device__ cartCoord cartCoord::operator*(const cartCoord &rhs) const {
    cartCoord prod;
    prod.coords[0] = coords[1]*rhs.coords[2]-coords[2]*rhs.coords[1];
    prod.coords[1] = coords[2]*rhs.coords[0]-coords[0]*rhs.coords[2];
    prod.coords[2] = coords[0]*rhs.coords[1]-coords[1]*rhs.coords[0];
    return prod;
}

__host__ __device__ float dotProd(const cartCoord &p1,const cartCoord &p2) {
    return p1.coords[0]*p2.coords[0]+p1.coords[1]*p2.coords[1]+p1.coords[2]*p2.coords[2];
}

__host__ __device__ cartCoord numDvd(const cartCoord &pnt, const float lambda) {
    if(lambda == 0) {
        printf("divisor cannot be 0.\n");
        return cartCoord(0,0,0);
    } else {
        return cartCoord(pnt.coords[0]/lambda,pnt.coords[1]/lambda,pnt.coords[2]/lambda);
    }
}

__host__ __device__ cartCoord numMul(const float lambda, const cartCoord &pnt) {
    return cartCoord(lambda*pnt.coords[0],lambda*pnt.coords[1],lambda*pnt.coords[2]);
}

__host__ __device__ float cartCoord::nrm2() const {
    return sqrtf(powf(coords[0],2)+powf(coords[1],2)+powf(coords[2],2));
}

__host__ __device__ float r(const cartCoord p1, const cartCoord p2) {
    cartCoord temp = p1-p2;
    return temp.nrm2();
}

__host__ __device__ float prpn2(const cartCoord n, const cartCoord p1, const cartCoord p2) {
    return ((p1.coords[0]-p2.coords[0])*n.coords[0]+(p1.coords[1]-p2.coords[1])*n.coords[1]
            +(p1.coords[2]-p2.coords[2])*n.coords[2])/r(p1,p2);
}

__host__ __device__ float prRpn2(const cartCoord n, const cartCoord p1, const cartCoord p2) {
    float temp1 = 1.0/powf(r(p1,p2),2), temp2 = prpn2(n,p1,p2);
    return -temp1*temp2;
}

__host__ __device__ cuFloatComplex green2(const float k, const cartCoord x, const cartCoord y) {
    cartCoord temp = x-y;
    float r = temp.nrm2();
    return green(k,r);
}

__host__ __device__ float PsiL2(const cartCoord p1, const cartCoord p2) {
    cartCoord temp = p1-p2;
    return PsiL(temp.nrm2());
}

__host__ __device__ float pPsiLpn2(const cartCoord n, const cartCoord p1, 
        const cartCoord p2) {
    return 1.0/(4*PI)*prRpn2(n,p1,p2);
}

__host__ __device__ cuFloatComplex pGpn2(const float k, const cartCoord n, 
        const cartCoord p1, const cartCoord p2) {
    cuFloatComplex temp1 = green2(k,p1,p2), temp2 = make_cuFloatComplex(-1.0/r(p1,p2),k);
    cuFloatComplex temp3 = cuCmulf(temp1,temp2);
    float temp4 = prpn2(n,p1,p2);
    return make_cuFloatComplex(temp4*cuCrealf(temp3),temp4*cuCimagf(temp3));
}

__device__ cuFloatComplex g_l1_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_1(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g)); //the imag part
}

__device__ cuFloatComplex g_l2_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_2(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g)); //the imag part
}

__device__ cuFloatComplex g_l3_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_3(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g)); //the imag part
}

__device__ cuFloatComplex h_l1_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_1(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
}

__device__ cuFloatComplex h_l2_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_2(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
} 

__device__ cuFloatComplex h_l3_nsgl(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_3(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
} 

__device__ cuFloatComplex g_l1_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = 1-rho;
    xi2 = rho*(1-theta);
    N = N_1(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g));
}

__device__ cuFloatComplex g_l2_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = 1-rho;
    xi2 = rho*(1-theta);
    N = N_2(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g));
}

__device__ cuFloatComplex g_l3_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = 1-rho;
    xi2 = rho*(1-theta);
    N = N_3(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g));
}

__device__ cuFloatComplex g_l1_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*theta;
    xi2 = 1-rho;
    N = N_1(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g));
}

__device__ cuFloatComplex g_l2_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*theta;
    xi2 = 1-rho;
    N = N_2(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g));
}

__device__ cuFloatComplex g_l3_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*theta;
    xi2 = 1-rho;
    N = N_3(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g));
}

__device__ cuFloatComplex g_l1_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_1(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g));
}

__device__ cuFloatComplex g_l2_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_2(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g));
}

__device__ cuFloatComplex g_l3_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp, omega = k*speed;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, crossProd;
    cuFloatComplex g;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_3(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    g = green2(k,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd*density*omega;
    return make_cuFloatComplex(-temp*cuCimagf(g),temp*cuCrealf(g));
}

__device__ cuFloatComplex h_l1_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = 1-rho;
    xi2 = rho*(1-theta);
    N = N_1(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
}

__device__ cuFloatComplex h_l2_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = 1-rho;
    xi2 = rho*(1-theta);
    N = N_2(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
}

__device__ cuFloatComplex h_l3_sgl1(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = 1-rho;
    xi2 = rho*(1-theta);
    N = N_3(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
}

__device__ cuFloatComplex h_l1_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*theta;
    xi2 = 1-rho;
    N = N_1(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
}

__device__ cuFloatComplex h_l2_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*theta;
    xi2 = 1-rho;
    N = N_2(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
}

__device__ cuFloatComplex h_l3_sgl2(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*theta;
    xi2 = 1-rho;
    N = N_3(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
}

__device__ cuFloatComplex h_l1_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_1(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
}

__device__ cuFloatComplex h_l2_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_2(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
}

__device__ cuFloatComplex h_l3_sgl3(const float k, const cartCoord x, 
        const cartCoord p1, const cartCoord p2, const cartCoord p3, 
        const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, N, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    cuFloatComplex gDrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    N = N_3(cartCoord2D(xi1,xi2));
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    gDrv = pGpn2(k,nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*N*vertCrossProd;
    return make_cuFloatComplex(temp*cuCrealf(gDrv),temp*cuCimagf(gDrv));
}

__device__ float c_l_nsgl(const cartCoord x, const cartCoord p1, const cartCoord p2, 
        const cartCoord p3, const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    float psiLdrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    psiLdrv = pPsiLpn2(nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd*psiLdrv;
    return temp;
}

__device__ float c_l_sgl1(const cartCoord x, const cartCoord p1, const cartCoord p2, 
        const cartCoord p3, const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    float psiLdrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = 1-rho;
    xi2 = rho*(1-theta);
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    psiLdrv = pPsiLpn2(nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd*psiLdrv;
    return temp;
}

__device__ float c_l_sgl2(const cartCoord x, const cartCoord p1, const cartCoord p2, 
        const cartCoord p3, const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    float psiLdrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*theta;
    xi2 = 1-rho;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    psiLdrv = pPsiLpn2(nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd*psiLdrv;
    return temp;
}

__device__ float c_l_sgl3(const cartCoord x, const cartCoord p1, const cartCoord p2, 
        const cartCoord p3, const int n, const int m) {
    float xi1, xi2, rho, theta, vertCrossProd, temp;
    float eta1 = INTPNTS[m], eta2 = INTPNTS[n];
    cartCoord y, nrml, crossProd;
    float psiLdrv;
    rho = 0.5+0.5*eta1;
    theta = 0.5+0.5*eta2;
    xi1 = rho*(1-theta);
    xi2 = rho*theta;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    nrml = crossProd.nrmlzd();
    y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
    psiLdrv = pPsiLpn2(nrml,x,y);
    temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd*psiLdrv;
    return temp;
}

//hCoeffs is of size 3*numPnts and cCoeffs is of size numPnts
__global__ void pntsElem_lnm_nsgl(const float k, const int l, const int n, const int m, 
        const triElem *elems, const cartCoord *pnts, const int numPnts, 
        cuFloatComplex *hCoeffs, cuFloatComplex *gCoeffs, float *cCoeffs) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numPnts) {
        triElem elem = elems[l]; //get the current element
  
        hCoeffs[3*idx] = h_l1_nsgl(k,pnts[idx],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        hCoeffs[3*idx+1] = h_l2_nsgl(k,pnts[idx],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        hCoeffs[3*idx+2] = h_l3_nsgl(k,pnts[idx],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        
        gCoeffs[3*idx] = g_l1_nsgl(k,pnts[idx],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        gCoeffs[3*idx+1] = g_l2_nsgl(k,pnts[idx],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        gCoeffs[3*idx+2] = g_l3_nsgl(k,pnts[idx],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        
        cCoeffs[idx] = c_l_nsgl(pnts[idx],pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
                n,m);
    }
    //Singularity integral has to be eliminated on the CPU end!
}

//gCoeffs and hCoeffs are of size 3*numElems, cCoeffs is of size numElems
__global__ void pntsElems_nm_sgl(const float k, const int n, const int m, const triElem *elems, 
        const int numElems, const cartCoord *pnts, cuFloatComplex *hCoeffs_sgl1, 
        cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, cuFloatComplex *gCoeffs_sgl1, 
        cuFloatComplex *gCoeffs_sgl2, cuFloatComplex *gCoeffs_sgl3, float *cCoeffs_sgl1, 
        float *cCoeffs_sgl2, float *cCoeffs_sgl3) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numElems) {
        triElem elem = elems[idx];
        
        hCoeffs_sgl1[3*idx] = h_l1_sgl1(k,pnts[elem.nodes[0]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        hCoeffs_sgl1[3*idx+1] = h_l2_sgl1(k,pnts[elem.nodes[0]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        hCoeffs_sgl1[3*idx+2] = h_l3_sgl1(k,pnts[elem.nodes[0]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        
        hCoeffs_sgl2[3*idx] = h_l1_sgl2(k,pnts[elem.nodes[1]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        hCoeffs_sgl2[3*idx+1] = h_l2_sgl2(k,pnts[elem.nodes[1]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        hCoeffs_sgl2[3*idx+2] = h_l3_sgl2(k,pnts[elem.nodes[1]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        
        hCoeffs_sgl3[3*idx] = h_l1_sgl3(k,pnts[elem.nodes[2]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        hCoeffs_sgl3[3*idx+1] = h_l2_sgl3(k,pnts[elem.nodes[2]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        hCoeffs_sgl3[3*idx+2] = h_l3_sgl3(k,pnts[elem.nodes[2]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        
        gCoeffs_sgl1[3*idx] = g_l1_sgl1(k,pnts[elem.nodes[0]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        gCoeffs_sgl1[3*idx+1] = g_l2_sgl1(k,pnts[elem.nodes[0]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        gCoeffs_sgl1[3*idx+2] = g_l3_sgl1(k,pnts[elem.nodes[0]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        
        gCoeffs_sgl2[3*idx] = g_l1_sgl2(k,pnts[elem.nodes[1]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        gCoeffs_sgl2[3*idx+1] = g_l2_sgl2(k,pnts[elem.nodes[1]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        gCoeffs_sgl2[3*idx+2] = g_l3_sgl2(k,pnts[elem.nodes[1]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        
        gCoeffs_sgl3[3*idx] = g_l1_sgl3(k,pnts[elem.nodes[2]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        gCoeffs_sgl3[3*idx+1] = g_l2_sgl3(k,pnts[elem.nodes[2]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        gCoeffs_sgl3[3*idx+2] = g_l3_sgl3(k,pnts[elem.nodes[2]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        
        cCoeffs_sgl1[idx] = c_l_sgl1(pnts[elem.nodes[0]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        
        cCoeffs_sgl2[idx] = c_l_sgl2(pnts[elem.nodes[1]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
        
        cCoeffs_sgl3[idx] = c_l_sgl3(pnts[elem.nodes[2]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],
                pnts[elem.nodes[2]],n,m);
    }
}

//Updated in an n and m loop
__global__ void updateSystemLhs_hg_nsgl(cuFloatComplex *A, const int numPnts, const int numCHIEF, 
        const int lda, const cuFloatComplex *hCoeffs, const cuFloatComplex *gCoeffs, 
        const triElem *elems, const int l) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numPnts+numCHIEF) {
        triElem elem = elems[l];
        cuFloatComplex hContrs[3], gContrs[3]; 
        cuFloatComplex pContrs[3], bc;
        
        bc = cuCdivf(elem.bc[0],elem.bc[1]);
        
        hContrs[0] = hCoeffs[3*idx];
        hContrs[1] = hCoeffs[3*idx+1];
        hContrs[2] = hCoeffs[3*idx+2];
        
        gContrs[0] = gCoeffs[3*idx];
        gContrs[1] = gCoeffs[3*idx+1];
        gContrs[2] = gCoeffs[3*idx+2];
        
        pContrs[0] = cuCsubf(hContrs[0],cuCmulf(bc,gContrs[0]));
        pContrs[1] = cuCsubf(hContrs[1],cuCmulf(bc,gContrs[1]));
        pContrs[2] = cuCsubf(hContrs[2],cuCmulf(bc,gContrs[2]));
        
        A[IDXC0(idx,elem.nodes[0],lda)] = cuCaddf(A[IDXC0(idx,elem.nodes[0],lda)],pContrs[0]);
        A[IDXC0(idx,elem.nodes[1],lda)] = cuCaddf(A[IDXC0(idx,elem.nodes[1],lda)],pContrs[1]);
        A[IDXC0(idx,elem.nodes[2],lda)] = cuCaddf(A[IDXC0(idx,elem.nodes[2],lda)],pContrs[2]);
    }
}

//numPnts does not include CHIEF points 
//Updated in an n and m loop
__global__ void updateSystemLhs_c_nsgl(cuFloatComplex *A, const int numPnts, const int lda, 
        const float *cCoeffs) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numPnts) {
        A[IDXC0(idx,idx,lda)] = cuCsubf(A[IDXC0(idx,idx,lda)],make_cuFloatComplex(cCoeffs[idx],0));
    }
}

//Updated in an n and m loop
__global__ void updateSystemRhs_nsgl(cuFloatComplex *B, const int numPnts, const int numCHIEF, 
        const int ldb, const int srcIdx, const cuFloatComplex *gCoeffs, const triElem *elems, 
        const int l) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numPnts+numCHIEF) {
        cuFloatComplex gContrs[3], bc, temp;
        gContrs[0] = gCoeffs[3*idx];
        gContrs[1] = gCoeffs[3*idx+1];
        gContrs[2] = gCoeffs[3*idx+2];
        
        triElem elem = elems[l];
        bc = cuCdivf(elem.bc[2],elem.bc[1]);
        
        temp = cuCmulf(bc,gContrs[0]);
        B[IDXC0(idx,srcIdx,ldb)] = cuCsubf(B[IDXC0(idx,srcIdx,ldb)],temp);
        temp = cuCmulf(bc,gContrs[1]);
        B[IDXC0(idx,srcIdx,ldb)] = cuCsubf(B[IDXC0(idx,srcIdx,ldb)],temp);
        temp = cuCmulf(bc,gContrs[2]);
        B[IDXC0(idx,srcIdx,ldb)] = cuCsubf(B[IDXC0(idx,srcIdx,ldb)],temp);
    }
}


//Updated in an n and m loop
__global__ void updateSystemLhs_hg_sgl(cuFloatComplex *A, const int lda, 
        cuFloatComplex *hCoeffs_sgl1, cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, 
        cuFloatComplex *gCoeffs_sgl1, cuFloatComplex *gCoeffs_sgl2, cuFloatComplex *gCoeffs_sgl3, 
        const triElem *elems, const int numElems) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numElems) {
        triElem elem = elems[idx];
        cuFloatComplex hContrs_sgl1[3], hContrs_sgl2[3], hContrs_sgl3[3], 
                gContrs_sgl1[3], gContrs_sgl2[3], gContrs_sgl3[3], 
                pContrs_sgl1[3], pContrs_sgl2[3], pContrs_sgl3[3];
        cuFloatComplex temp, bc = cuCdivf(elem.bc[0],elem.bc[1]);
        
        //singular 1:
        hContrs_sgl1[0] = hCoeffs_sgl1[3*idx];
        hContrs_sgl1[1] = hCoeffs_sgl1[3*idx+1];
        hContrs_sgl1[2] = hCoeffs_sgl1[3*idx+2];
        
        gContrs_sgl1[0] = gCoeffs_sgl1[3*idx];
        gContrs_sgl1[1] = gCoeffs_sgl1[3*idx+1];
        gContrs_sgl1[2] = gCoeffs_sgl1[3*idx+2];
        
        temp = cuCmulf(bc,gContrs_sgl1[0]);
        pContrs_sgl1[0] = cuCsubf(hContrs_sgl1[0],temp);
        temp = cuCmulf(bc,gContrs_sgl1[1]);
        pContrs_sgl1[1] = cuCsubf(hContrs_sgl1[1],temp);
        temp = cuCmulf(bc,gContrs_sgl1[2]);
        pContrs_sgl1[2] = cuCsubf(hContrs_sgl1[2],temp);
        
        A[IDXC0(elem.nodes[0],elem.nodes[0],lda)] = cuCaddf(A[IDXC0(elem.nodes[0],elem.nodes[0],lda)],
                pContrs_sgl1[0]);
        A[IDXC0(elem.nodes[0],elem.nodes[1],lda)] = cuCaddf(A[IDXC0(elem.nodes[0],elem.nodes[1],lda)],
                pContrs_sgl1[1]);
        A[IDXC0(elem.nodes[0],elem.nodes[2],lda)] = cuCaddf(A[IDXC0(elem.nodes[0],elem.nodes[2],lda)],
                pContrs_sgl1[2]);
        
        //singular 2:
        hContrs_sgl2[0] = hCoeffs_sgl2[3*idx];
        hContrs_sgl2[1] = hCoeffs_sgl2[3*idx+1];
        hContrs_sgl2[2] = hCoeffs_sgl2[3*idx+2];
        
        gContrs_sgl2[0] = gCoeffs_sgl2[3*idx];
        gContrs_sgl2[1] = gCoeffs_sgl2[3*idx+1];
        gContrs_sgl2[2] = gCoeffs_sgl2[3*idx+2];
        
        temp = cuCmulf(bc,gContrs_sgl2[0]);
        pContrs_sgl2[0] = cuCsubf(hContrs_sgl2[0],temp);
        temp = cuCmulf(bc,gContrs_sgl2[1]);
        pContrs_sgl2[1] = cuCsubf(hContrs_sgl2[1],temp);
        temp = cuCmulf(bc,gContrs_sgl1[2]);
        pContrs_sgl2[2] = cuCsubf(hContrs_sgl2[2],temp);
        
        A[IDXC0(elem.nodes[1],elem.nodes[0],lda)] = cuCaddf(A[IDXC0(elem.nodes[1],elem.nodes[0],lda)],
                pContrs_sgl2[0]);
        A[IDXC0(elem.nodes[1],elem.nodes[1],lda)] = cuCaddf(A[IDXC0(elem.nodes[1],elem.nodes[1],lda)],
                pContrs_sgl2[1]);
        A[IDXC0(elem.nodes[1],elem.nodes[2],lda)] = cuCaddf(A[IDXC0(elem.nodes[1],elem.nodes[2],lda)],
                pContrs_sgl2[2]);
        
        //singular 2:
        hContrs_sgl3[0] = hCoeffs_sgl3[3*idx];
        hContrs_sgl3[1] = hCoeffs_sgl3[3*idx+1];
        hContrs_sgl3[2] = hCoeffs_sgl3[3*idx+2];
        
        gContrs_sgl3[0] = gCoeffs_sgl3[3*idx];
        gContrs_sgl3[1] = gCoeffs_sgl3[3*idx+1];
        gContrs_sgl3[2] = gCoeffs_sgl3[3*idx+2];
        
        temp = cuCmulf(bc,gContrs_sgl3[0]);
        pContrs_sgl3[0] = cuCsubf(hContrs_sgl3[0],temp);
        temp = cuCmulf(bc,gContrs_sgl3[1]);
        pContrs_sgl3[1] = cuCsubf(hContrs_sgl3[1],temp);
        temp = cuCmulf(bc,gContrs_sgl3[2]);
        pContrs_sgl3[2] = cuCsubf(hContrs_sgl3[2],temp);
        
        A[IDXC0(elem.nodes[2],elem.nodes[0],lda)] = cuCaddf(A[IDXC0(elem.nodes[2],elem.nodes[0],lda)],
                pContrs_sgl3[0]);
        A[IDXC0(elem.nodes[2],elem.nodes[1],lda)] = cuCaddf(A[IDXC0(elem.nodes[2],elem.nodes[1],lda)],
                pContrs_sgl3[1]);
        A[IDXC0(elem.nodes[2],elem.nodes[2],lda)] = cuCaddf(A[IDXC0(elem.nodes[2],elem.nodes[2],lda)],
                pContrs_sgl3[2]);
    }
}

__global__ void updateSystemLhs_c_sgl(cuFloatComplex *A, const int lda, float *cCoeffs_sgl1, 
        float *cCoeffs_sgl2, float *cCoeffs_sgl3, const triElem *elems, const int numElems) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numElems) {
        triElem elem = elems[idx];
        float cContr_sgl1, cContr_sgl2, cContr_sgl3;
        
        cContr_sgl1 = cCoeffs_sgl1[idx];
        A[IDXC0(elem.nodes[0],elem.nodes[0],lda)] = cuCsubf(A[IDXC0(elem.nodes[0],elem.nodes[0],lda)],
                make_cuFloatComplex(cContr_sgl1,0));
        
        cContr_sgl2 = cCoeffs_sgl2[idx];
        A[IDXC0(elem.nodes[1],elem.nodes[1],lda)] = cuCsubf(A[IDXC0(elem.nodes[1],elem.nodes[1],lda)],
                make_cuFloatComplex(cContr_sgl2,0));
        
        cContr_sgl3 = cCoeffs_sgl3[idx];
        A[IDXC0(elem.nodes[2],elem.nodes[2],lda)] = cuCsubf(A[IDXC0(elem.nodes[2],elem.nodes[2],lda)],
                make_cuFloatComplex(cContr_sgl3,0));
    }
}



__host__ __device__ float trnglArea(const cartCoord p1, const cartCoord p2) {
    cartCoord temp = p1*p2;
    return temp.nrm2()/2.0;
}

__host__ __device__ cartCoord cartCoord::nrmlzd() {
    if(nrm2() == 0) {
        return cartCoord(nanf(""),nanf(""),nanf(""));
    } else {
        float nrm = nrm2();
        return numDvd(*this,nrm);
    }
}

__host__ __device__ cartCoord rayPlaneInt(const cartCoord sp, const cartCoord dir,
    const cartCoord n, const cartCoord pnt) {
    cartCoord temp;
    if(dotProd(n,sp-pnt) == 0) {
        temp =  sp;
    } else {
        if(dotProd(n,dir)==0) {
            temp =  cartCoord(nanf(""),nanf(""),nanf(""));
        } else {
            float t = (dotProd(n,pnt)-dotProd(n,sp))/dotProd(n,dir);
            if(t>0) {
                temp = sp+numMul(t,dir);
            } else {
                temp = cartCoord(nanf(""),nanf(""),nanf(""));
            }
        }
    }
    //printf("(%f,%f,%f)\n",temp.coords[0],temp.coords[1],temp.coords[2]);
    return temp;
}

__host__ __device__ bool cartCoord::isLegal() const {
    if(coords[0]!=coords[0]||coords[1]!=coords[1]||coords[2]!=coords[2]) {
        return false;
    } else {
        return true;
    }
}

__host__ __device__ bool cartCoord::isEqual(const cartCoord p) const {
    if(abs(coords[0]-p.coords[0])<EPS && abs(coords[1]-p.coords[1])<EPS 
            && abs(coords[2]-p.coords[2])<EPS) {
        return true;
    } else {
        return false;
    }
}


__host__ __device__ bool cartCoord::isInsideTrngl(const cartCoord p1,
        const cartCoord p2, const cartCoord p3) const {
    if(!isLegal()) {
        return false;
    } else {
        cartCoord v12 = p2-p1, v23 = p3-p2, v31 = p1-p3;
        cartCoord v1p = *this-p1, v2p = *this-p2, v3p = *this-p3;
        cartCoord t1 = v12*v1p, t2 = v23*v2p, t3 = v31*v3p;
        cartCoord t1Nrm = t1.nrmlzd(), t2Nrm = t2.nrmlzd(), t3Nrm = t3.nrmlzd();
        if(t1Nrm.isEqual(t2Nrm) && t2Nrm.isEqual(t3Nrm)) {
            return true;
        } else {
            return false;
        }
    }
}

__host__ __device__ bool rayTrnglInt(const cartCoord sp, const cartCoord dir,
    const cartCoord p1, const cartCoord p2, const cartCoord p3) {
    cartCoord n = (p2-p1)*(p3-p1);
    cartCoord intPnt = rayPlaneInt(sp,dir,n,p1);
    return intPnt.isInsideTrngl(p1,p2,p3);
}

//triangular element class
__host__ __device__ triElem::triElem(const triElem &rhs) {
    for(int i=0;i<3;i++) {
        nodes[i] = rhs.nodes[i];
        bc[i] = rhs.bc[i];
    }
}

__host__ __device__ triElem& triElem::operator=(const triElem &rhs) {
    for(int i=0;i<3;i++) {
        nodes[i] = rhs.nodes[i];
        bc[i] = rhs.bc[i];
    }
    return *this;
}

ostream& operator<<(ostream &out, const triElem &rhs) {
    out << "nodes indices: " << rhs.nodes[0] << ", " << rhs.nodes[1] << ", " 
            << rhs.nodes[2] << " boundary condition: " << "A = " << rhs.bc[0] 
            << ", B = " << rhs.bc[1] << ", C = " << rhs.bc[2];
    return out;
}

//class mesh
int mesh::readObj(const char *file) {
    int temp[3];
    vector<cartCoord> pntVec; 
    vector<triElem> elemVec;
    cartCoord pnt;
    triElem elem;
    FILE *fp = fopen(file,"r");
    if (fp == NULL) {
        printf("Failed to open file.\n");
        return EXIT_FAILURE;
    }
    int i = 0;
    char line[50];
    char type[5];
    while (fgets(line,49,fp) != NULL) {
        if (line[0] == 'v') {
            sscanf(line,"%s %f %f %f",type,&pnt.coords[0],&pnt.coords[1],&pnt.coords[2]);
            pntVec.push_back(pnt);
        }

        if (line[0] == 'f') {
            sscanf(line,"%s %d %d %d",type,&temp[0],&temp[1],&temp[2]);
            elem.nodes[0] = temp[0]-1;
            elem.nodes[1] = temp[1]-1;
            elem.nodes[2] = temp[2]-1;
            elem.bc[0] = make_cuFloatComplex(0,0); // ca=0
            elem.bc[1] = make_cuFloatComplex(1,0); // cb=1
            elem.bc[2] = make_cuFloatComplex(0,0); // cc=0
            elemVec.push_back(elem);
        }
    }
    if(pnts != NULL) {
        delete[] pnts;
    }
    pnts = new cartCoord[pntVec.size()];
    for(i=0;i<pntVec.size();i++) {
        pnts[i] = pntVec[i];
    }
    numPnts = pntVec.size();
    
    if(elems != NULL) {
        delete[] elems;
    }
    elems = new triElem[elemVec.size()];
    for(i=0;i<elemVec.size();i++) {
        elems[i] = elemVec[i];
    }
    numElems = elemVec.size();
    fclose(fp);
    return EXIT_SUCCESS;
}

mesh::mesh(const mesh &rhs) {
    if(rhs.numPnts > 0) {
        if(pnts != NULL) {
            delete[] pnts;
        }
        numPnts = rhs.numPnts;
        pnts = new cartCoord[numPnts];
        for(int i=0;i<numPnts;i++) {
            pnts[i] = rhs.pnts[i];
        }
    } else {
        if(pnts != NULL) {
            delete[] pnts;
        }
    }
    
    if(rhs.numElems > 0) {
        if(elems != NULL) {
            delete[] elems;
        }
        numElems = rhs.numElems;
        elems = new triElem[numElems];
        for(int i=0;i<numPnts;i++) {
            elems[i] = rhs.elems[i];
        }
    } else {
        if(elems != NULL) {
            delete[] elems;
        }
    }
}

mesh::~mesh() {
    if(pnts != NULL) {
        delete[] pnts;
    }
    if(elems != NULL) {
        delete[] elems;
    }
    if(chiefPnts != NULL) {
        delete[] chiefPnts;
    }
}

mesh& mesh::operator=(const mesh &rhs) {
    if(rhs.numPnts > 0) {
        if(pnts != NULL) {
            delete[] pnts;
        }
        numPnts = rhs.numPnts;
        pnts = new cartCoord[numPnts];
        for(int i=0;i<numPnts;i++) {
            pnts[i] = rhs.pnts[i];
        }
    } else {
        if(pnts != NULL) {
            delete[] pnts;
        }
    }
    
    if(rhs.numElems > 0) {
        if(elems != NULL) {
            delete[] elems;
        }
        numElems = rhs.numElems;
        elems = new triElem[numElems];
        for(int i=0;i<numPnts;i++) {
            elems[i] = rhs.elems[i];
        }
    } else {
        if(elems != NULL) {
            delete[] elems;
        }
    }
    return *this;
}

void mesh::printBB() {
    cout << "Lower x: " << xl << "Higher x: " << xu << "Lower y: " << yl 
            << "Higher y: " << yu << "Lower z: " << zl << "Higher z: " << zu << endl;
}

ostream& operator<<(ostream &out, const mesh &rhs) {
    for(int i=0;i<rhs.numPnts;i++) {
        cout << rhs.pnts[i] << endl;
    }
    
    for(int i=0;i<rhs.numElems;i++) {
        cout << rhs.elems[i] << endl;
    }
    
    return out;
}

int mesh::findBB(const float threshold) {
    if(numPnts!=0 && numElems!=0) {
        xl = pnts[0].coords[0]; 
        xu = pnts[0].coords[0]; 
        yl = pnts[0].coords[1]; 
        yu = pnts[0].coords[1];
        zl = pnts[0].coords[2]; 
        zu = pnts[0].coords[2];
        for(int i=1;i<numPnts;i++) {
            if(pnts[i].coords[0] < xl) {
                xl = pnts[i].coords[0];
            }
            if(pnts[i].coords[0] > xu) {
                xu = pnts[i].coords[0];
            }
            if(pnts[i].coords[1] < yl) {
                yl = pnts[i].coords[1];
            }
            if(pnts[i].coords[1] > yu) {
                yu = pnts[i].coords[1];
            }
            if(pnts[i].coords[2] < zl) {
                zl = pnts[i].coords[2];
            }
            if(pnts[i].coords[2] > zu) {
                zu = pnts[i].coords[2];
            }
        }
        xl-=threshold;
        xu+=threshold;
        yl-=threshold;
        yu+=threshold;
        zl-=threshold;
        zu+=threshold;
        
        return EXIT_SUCCESS;
    } else {
        cout << "Not enough mesh information!" << endl;
        return EXIT_FAILURE;
    }
}

__global__ void rayTrnglsInt(const cartCoord sp, const cartCoord dir, 
        const cartCoord *pnts, const triElem *elems, const int numElems, bool *flags) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; 
    if(idx < numElems) {
        flags[idx] = rayTrnglInt(sp,dir,pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]]);
        //printf("%d: %d\n",idx,flags[idx]);
    }
    
}

__global__ void distPntPnts(const cartCoord sp, const cartCoord *pnts, const int numPnts, float *dists) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; 
    if(idx < numPnts) {
        cartCoord temp = pnts[idx]-sp;
        dists[idx] = temp.nrm2();
    }
}

int mesh::genCHIEF(const int num, const float threshold) {
    numCHIEF = num;
    if(chiefPnts != NULL) {
        delete[] chiefPnts;
    }
    chiefPnts = new cartCoord[numCHIEF];
    float randNums[3];
    int width = 32, numBlocks;
    float xRand, yRand, zRand;
    unsigned long long seed = time(0);
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGeneratorHost(&gen,CURAND_RNG_PSEUDO_DEFAULT));
    cartCoord sp;
    cartCoord *pnts_d;
    triElem *elems_d;
    HOST_CALL(meshCloudToGPU(&pnts_d,&elems_d));
    float *dists = new float[numPnts];
    float *dists_d;
    CUDA_CALL(cudaMalloc(&dists_d,numPnts*sizeof(float)));
    bool *flags_d;
    CUDA_CALL(cudaMalloc(&flags_d,numElems*sizeof(bool)));
    bool *flags = new bool[numElems];
    int cnt = 0; //counter for number of CHIEF points
    while(cnt < numCHIEF) {
        do {
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,seed++));
            CURAND_CALL(curandGenerateUniform(gen,randNums,3));
            //printf("(%f,%f,%f)\n",randNums[0],randNums[1],randNums[2]);
            dirCHIEF.set(randNums[0],randNums[1],randNums[2]);
            xRand = descale(xl,xu,randNums[0]);
            yRand = descale(yl,yu,randNums[1]);
            zRand = descale(zl,zu,randNums[2]);
            sp.set(xRand,yRand,zRand);
            numBlocks = (numElems+width-1)/width;
            rayTrnglsInt<<<numBlocks,width>>>(sp,dirCHIEF,pnts_d,elems_d,numElems,flags_d);
            CUDA_CALL(cudaMemcpy(flags,flags_d,numElems*sizeof(bool),cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
            numBlocks = (numPnts+width-1)/width;
            distPntPnts<<<numBlocks,width>>>(sp,pnts_d,numPnts,dists_d);
            CUDA_CALL(cudaMemcpy(dists,dists_d,numPnts*sizeof(float),cudaMemcpyDeviceToHost));
            //printf("Minimum distance: %f\n",arrayMin(dists,numPnts));
            
            //cout << inObj(flags,numElems) << endl;
        } while((!inObj(flags,numElems))||(arrayMin(dists,numPnts)<threshold));
        chiefPnts[cnt].set(xRand,yRand,zRand);
        cnt++;
    }
    
    delete[] dists;
    delete[] flags;
    CUDA_CALL(cudaFree(pnts_d));
    CUDA_CALL(cudaFree(elems_d));
    CUDA_CALL(cudaFree(flags_d));
    CUDA_CALL(cudaFree(dists_d));
    CURAND_CALL(curandDestroyGenerator(gen));
    
    for(int i=0;i<numCHIEF;i++) {
        cout << chiefPnts[i] << endl;
    }
     
    return EXIT_SUCCESS;
}

int mesh::meshCloudToGPU(cartCoord **pPnts_d,triElem **pElems_d) {
    if(pnts!=NULL && elems!=NULL) {
        CUDA_CALL(cudaMalloc(pPnts_d,numPnts*sizeof(cartCoord)));
        CUDA_CALL(cudaMemcpy(*pPnts_d,pnts,numPnts*sizeof(cartCoord),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc(pElems_d,numElems*sizeof(triElem)));
        CUDA_CALL(cudaMemcpy(*pElems_d,elems,numElems*sizeof(triElem),cudaMemcpyHostToDevice));
    }
    
    return EXIT_SUCCESS;
}

int mesh::chiefToGPU(cartCoord **pchiefPnts) {
    if(numCHIEF!=0) {
        CUDA_CALL(cudaMalloc(pchiefPnts,numCHIEF*sizeof(cartCoord)));
        CUDA_CALL(cudaMemcpy(*pchiefPnts,chiefPnts,numCHIEF*sizeof(cartCoord),cudaMemcpyHostToDevice));
    }
    return EXIT_SUCCESS;
}

int mesh::meshToGPU(cartCoord **pPnts_d, triElem **pElems_d) {
    if(numPnts==0 || numElems==0 || numCHIEF==0) {
        cout << "The mesh object is incomplete." << endl;
        return EXIT_FAILURE;
    } else {
        int i;
        cartCoord *pnts_h = new cartCoord[numPnts+numCHIEF];
        for(i=0;i<numPnts;i++) {
            pnts_h[i] = pnts[i];
        }
        for(i=0;i<numCHIEF;i++) {
            pnts_h[numPnts+i] = chiefPnts[i];
        }
        CUDA_CALL(cudaMalloc(pPnts_d,(numPnts+numCHIEF)*sizeof(cartCoord)));
        CUDA_CALL(cudaMemcpy(pPnts_d,pnts_h,(numPnts+numCHIEF)*sizeof(cartCoord),cudaMemcpyHostToDevice));
        
        CUDA_CALL(cudaMalloc(pElems_d,numElems*sizeof(triElem)));
        CUDA_CALL(cudaMemcpy(pElems_d,elems,numElems*sizeof(triElem),cudaMemcpyHostToDevice));
        return EXIT_SUCCESS;
    }
}

//cartCoord2D
__host__ __device__ cartCoord2D::cartCoord2D(const cartCoord2D &rhs) {
    coords[0] = rhs.coords[0];
    coords[1] = rhs.coords[1];
}

__host__ __device__ cartCoord2D& cartCoord2D::operator=(const cartCoord2D &rhs) {
    coords[0] = rhs.coords[0];
    coords[1] = rhs.coords[1];
    return *this;
}

__host__ __device__ void cartCoord2D::set(const float x, const float y) {
    coords[0] = x;
    coords[1] = y;
}

__host__ __device__ cartCoord2D cartCoord2D::operator+(const cartCoord2D &rhs) const {
    cartCoord2D temp;
    temp.coords[0] = coords[0] + rhs.coords[0];
    temp.coords[1] = coords[1] + rhs.coords[1];
    return temp;
}

__host__ __device__ cartCoord2D cartCoord2D::operator-(const cartCoord2D &rhs) const {
    cartCoord2D temp;
    temp.coords[0] = coords[0] - rhs.coords[0];
    temp.coords[1] = coords[1] - rhs.coords[1];
    return temp;
}

__host__ __device__ cartCoord2D numDvd(const cartCoord2D &dividend, 
        const float divisor) {
    if(divisor == 0) {
        printf("divisor cannot be 0.\n");
        return cartCoord2D(0,0);
    } else {
        return cartCoord2D(dividend.coords[0]/divisor,dividend.coords[1]/divisor);
    }
}

__host__ __device__ cartCoord2D numMul(const float lambda, const cartCoord2D &pnt) {
    return cartCoord2D(lambda*pnt.coords[0],lambda*pnt.coords[1]);
}

__host__ __device__ float Psi_L(const cartCoord pnt) {
    return 1.0/(4*PI*pnt.nrm2());
}

__host__ __device__ float N_1(const cartCoord2D pnt) {
    return pnt.coords[0];
}

__host__ __device__ float N_2(const cartCoord2D pnt) {
    return pnt.coords[1];
}

__host__ __device__ float N_3(const cartCoord2D pnt) {
    return 1-pnt.coords[0]-pnt.coords[1];
}

__host__ __device__ float pN1pXi1(const cartCoord2D pnt) {
    return 1.0;
}

__host__ __device__ float pN1pXi2(const cartCoord2D pnt) {
    return 0;
}

__host__ __device__ float pN2pXi1(const cartCoord2D pnt) {
    return 0;
}

__host__ __device__ float pN2pXi2(const cartCoord2D pnt) {
    return 1.0;
}

__host__ __device__ float pN3pXi1(const cartCoord2D pnt) {
    return -1.0;
}

__host__ __device__ float pN3pXi2(const cartCoord2D pnt) {
    return -1.0;
}

__host__ __device__ cartCoord xiToElem(const cartCoord pnt1, const cartCoord pnt2,
        const cartCoord pnt3, const cartCoord2D localPnt) {
    return numMul(N_1(localPnt),pnt1)+numMul(N_2(localPnt),pnt2)
            +numMul(N_3(localPnt),pnt3); 
}

__host__ __device__ cartCoord pRvpXi1TimespRvpXi2(const cartCoord pnt1, const cartCoord pnt2, 
        const cartCoord pnt3) {
    return (pnt1-pnt3)*(pnt2-pnt3);
}

__host__ __device__ cartCoord2D etaToRhoTheta(const cartCoord2D s) {
    cartCoord2D temp;
    temp.coords[0] = 0.5+0.5*s.coords[0];
    temp.coords[1] = 0.5+0.5*s.coords[1];
    return temp;
}

__host__ __device__ cartCoord2D rhoThetaToXi_3(const cartCoord2D s) {
    cartCoord2D temp;
    temp.coords[0] = s.coords[0]*(1-s.coords[1]); //rho*(1-theta)
    temp.coords[1] = s.coords[0]*s.coords[1];
    return temp;
}

__host__ __device__ cartCoord2D rhoThetaToXi_1(const cartCoord2D s) {
    cartCoord2D temp;
    temp.coords[0] = 1-s.coords[0];
    temp.coords[1] = s.coords[0]*(1-s.coords[1]);
    return temp;
}

__host__ __device__ cartCoord2D rhoThetaToXi_2(const cartCoord2D s) {
    cartCoord2D  temp;
    temp.coords[0] = s.coords[0]*s.coords[1];
    temp.coords[1] = 1-s.coords[0];
    return temp;
}

