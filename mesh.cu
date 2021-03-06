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
#include "atomicFuncs.h"
#include "GMRES.h"

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

std::ostream& operator<<(std::ostream &out,const cartCoord &rhs) {
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
    return sqrtf(coords[0]*coords[0]+coords[1]*coords[1]+coords[2]*coords[2]);
}

__host__ __device__ float r(const cartCoord p1, const cartCoord p2) {
    cartCoord temp = p1-p2;
    return temp.nrm2();
}

__host__ __device__ float prpn2(const cartCoord n, const cartCoord p1, const cartCoord p2) {
    return ((p2.coords[0]-p1.coords[0])*n.coords[0]+(p2.coords[1]-p1.coords[1])*n.coords[1]
            +(p2.coords[2]-p1.coords[2])*n.coords[2])/r(p1,p2);
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

__host__ __device__ cuFloatComplex pntSrc(const float k, const float q, 
        const cartCoord x, const cartCoord y) {
    cuFloatComplex temp = green2(k,x,y);
    return make_cuFloatComplex(q*cuCrealf(temp),q*cuCimagf(temp));
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
    cuFloatComplex temp1 = green2(k,p1,p2), temp2 = make_cuFloatComplex(-1.0/r(p1,p2),-k);
    cuFloatComplex temp3 = cuCmulf(temp1,temp2);
    float temp4 = prpn2(n,p1,p2);
    return make_cuFloatComplex(temp4*cuCrealf(temp3),temp4*cuCimagf(temp3));
}

__device__ void g_l_nsgl(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *gCoeff1, 
        cuFloatComplex *gCoeff2, cuFloatComplex *gCoeff3) {
    *gCoeff1 = make_cuFloatComplex(0,0);
    *gCoeff2 = make_cuFloatComplex(0,0);
    *gCoeff3 = make_cuFloatComplex(0,0);
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, N1, N2, N3, temp[4], 
            omega = k*speed;
    cartCoord y, crossProd;
    cuFloatComplex g;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = rho*(1-theta);
            xi2 = rho*theta;
            N1 = N_1(cartCoord2D(xi1,xi2));
            N2 = N_2(cartCoord2D(xi1,xi2));
            N3 = N_3(cartCoord2D(xi1,xi2));
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            g = green2(k,x,y);
            temp[0] = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd*density*omega;
            temp[1] = temp[0]*N1;
            temp[2] = temp[0]*N2;
            temp[3] = temp[0]*N3;
            *gCoeff1 = cuCaddf(*gCoeff1,make_cuFloatComplex(-temp[1]*cuCimagf(g),
                    temp[1]*cuCrealf(g)));
            *gCoeff2 = cuCaddf(*gCoeff2,make_cuFloatComplex(-temp[2]*cuCimagf(g),
                    temp[2]*cuCrealf(g)));
            *gCoeff3 = cuCaddf(*gCoeff3,make_cuFloatComplex(-temp[3]*cuCimagf(g),
                    temp[3]*cuCrealf(g)));
        }
    }
    printComplexMatrix(gCoeff1,1,1,1);
}

__device__ void h_l_nsgl(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *hCoeff1, 
        cuFloatComplex *hCoeff2, cuFloatComplex *hCoeff3) {
    *hCoeff1 = make_cuFloatComplex(0,0);
    *hCoeff2 = make_cuFloatComplex(0,0);
    *hCoeff3 = make_cuFloatComplex(0,0);
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, N1, N2, N3, temp[4];
    cartCoord y, crossProd, normal;
    cuFloatComplex g;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    normal = crossProd.nrmlzd();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = rho*(1-theta);
            xi2 = rho*theta;
            N1 = N_1(cartCoord2D(xi1,xi2));
            N2 = N_2(cartCoord2D(xi1,xi2));
            N3 = N_3(cartCoord2D(xi1,xi2));
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            g = pGpn2(k,normal,x,y);
            temp[0] = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd;
            temp[1] = temp[0]*N1;
            temp[2] = temp[0]*N2;
            temp[3] = temp[0]*N3;
            *hCoeff1 = cuCaddf(*hCoeff1,make_cuFloatComplex(temp[1]*cuCrealf(g),
                    temp[1]*cuCimagf(g)));
            *hCoeff2 = cuCaddf(*hCoeff2,make_cuFloatComplex(temp[2]*cuCrealf(g),
                    temp[2]*cuCimagf(g)));
            *hCoeff3 = cuCaddf(*hCoeff3,make_cuFloatComplex(temp[3]*cuCrealf(g),
                    temp[3]*cuCimagf(g)));
        }
    }
    //printComplexMatrix(hCoeff1,1,1,1);
}

__device__ cuFloatComplex h_l_1_nsgl(const float k, const int xIdx, const int nod1, 
        const int nod2, const int nod3, const cartCoord *pnts) {
    //Declaration of variables
    cuFloatComplex hCoeff = make_cuFloatComplex(0,0), g;
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, N1, temp;
    cartCoord y, crossProd, normal;
    int n, m;
    
    //Computation of normal and jacobian of the first transform
    crossProd = (pnts[nod1]-pnts[nod3])*(pnts[nod2]-pnts[nod3]);
    vertCrossProd = crossProd.nrm2();
    normal = crossProd.nrmlzd();
    
    //Accumulate the sum
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = rho*(1-theta);
            xi2 = rho*theta;
            N1 = N_1(cartCoord2D(xi1,xi2));
            y = xiToElem(pnts[nod1],pnts[nod2],pnts[nod3],cartCoord2D(xi1,xi2));
            g = pGpn2(k,normal,pnts[xIdx],y);
            temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd*N1;
            hCoeff = cuCaddf(hCoeff,make_cuFloatComplex(temp*cuCrealf(g),
                    temp*cuCimagf(g)));
        }
    }
    return hCoeff;
}

__device__ void g_l_sgl1(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *gCoeff1, 
        cuFloatComplex *gCoeff2, cuFloatComplex *gCoeff3) {
    *gCoeff1 = make_cuFloatComplex(0,0);
    *gCoeff2 = make_cuFloatComplex(0,0);
    *gCoeff3 = make_cuFloatComplex(0,0);
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, N1, N2, N3, temp[4], 
            omega = k*speed;
    cartCoord y, crossProd;
    cuFloatComplex g;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = 1-rho;
            xi2 = rho*(1-theta);
            N1 = N_1(cartCoord2D(xi1,xi2));
            N2 = N_2(cartCoord2D(xi1,xi2));
            N3 = N_3(cartCoord2D(xi1,xi2));
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            g = green2(k,x,y);
            temp[0] = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd*density*omega;
            temp[1] = temp[0]*N1;
            temp[2] = temp[0]*N2;
            temp[3] = temp[0]*N3;
            *gCoeff1 = cuCaddf(*gCoeff1,make_cuFloatComplex(-temp[1]*cuCimagf(g),
                    temp[1]*cuCrealf(g)));
            *gCoeff2 = cuCaddf(*gCoeff2,make_cuFloatComplex(-temp[2]*cuCimagf(g),
                    temp[2]*cuCrealf(g)));
            *gCoeff3 = cuCaddf(*gCoeff3,make_cuFloatComplex(-temp[3]*cuCimagf(g),
                    temp[3]*cuCrealf(g)));
        }
    }
}

__device__ void g_l_sgl2(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *gCoeff1, 
        cuFloatComplex *gCoeff2, cuFloatComplex *gCoeff3) {
    *gCoeff1 = make_cuFloatComplex(0,0);
    *gCoeff2 = make_cuFloatComplex(0,0);
    *gCoeff3 = make_cuFloatComplex(0,0);
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, N1, N2, N3, temp[4], 
            omega = k*speed;
    cartCoord y, crossProd;
    cuFloatComplex g;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = rho*theta;
            xi2 = 1-rho;
            N1 = N_1(cartCoord2D(xi1,xi2));
            N2 = N_2(cartCoord2D(xi1,xi2));
            N3 = N_3(cartCoord2D(xi1,xi2));
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            g = green2(k,x,y);
            temp[0] = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd*density*omega;
            temp[1] = temp[0]*N1;
            temp[2] = temp[0]*N2;
            temp[3] = temp[0]*N3;
            *gCoeff1 = cuCaddf(*gCoeff1,make_cuFloatComplex(-temp[1]*cuCimagf(g),
                    temp[1]*cuCrealf(g)));
            *gCoeff2 = cuCaddf(*gCoeff2,make_cuFloatComplex(-temp[2]*cuCimagf(g),
                    temp[2]*cuCrealf(g)));
            *gCoeff3 = cuCaddf(*gCoeff3,make_cuFloatComplex(-temp[3]*cuCimagf(g),
                    temp[3]*cuCrealf(g)));
        }
    }
}

__device__ void g_l_sgl3(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *gCoeff1, 
        cuFloatComplex *gCoeff2, cuFloatComplex *gCoeff3) {
    *gCoeff1 = make_cuFloatComplex(0,0);
    *gCoeff2 = make_cuFloatComplex(0,0);
    *gCoeff3 = make_cuFloatComplex(0,0);
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, N1, N2, N3, temp[4], 
            omega = k*speed;
    cartCoord y, crossProd;
    cuFloatComplex g;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = rho*(1-theta);
            xi2 = rho*theta;
            N1 = N_1(cartCoord2D(xi1,xi2));
            N2 = N_2(cartCoord2D(xi1,xi2));
            N3 = N_3(cartCoord2D(xi1,xi2));
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            g = green2(k,x,y);
            temp[0] = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd*density*omega;
            temp[1] = temp[0]*N1;
            temp[2] = temp[0]*N2;
            temp[3] = temp[0]*N3;
            *gCoeff1 = cuCaddf(*gCoeff1,make_cuFloatComplex(-temp[1]*cuCimagf(g),
                    temp[1]*cuCrealf(g)));
            *gCoeff2 = cuCaddf(*gCoeff2,make_cuFloatComplex(-temp[2]*cuCimagf(g),
                    temp[2]*cuCrealf(g)));
            *gCoeff3 = cuCaddf(*gCoeff3,make_cuFloatComplex(-temp[3]*cuCimagf(g),
                    temp[3]*cuCrealf(g)));
        }
    }
}

__device__ void h_l_sgl1(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *hCoeff1, 
        cuFloatComplex *hCoeff2, cuFloatComplex *hCoeff3) {
    *hCoeff1 = make_cuFloatComplex(0,0);
    *hCoeff2 = make_cuFloatComplex(0,0);
    *hCoeff3 = make_cuFloatComplex(0,0);
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, N1, N2, N3, temp[4];
    cartCoord y, crossProd, normal;
    cuFloatComplex g;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    normal = crossProd.nrmlzd();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = 1-rho;
            xi2 = rho*(1-theta);
            N1 = N_1(cartCoord2D(xi1,xi2));
            N2 = N_2(cartCoord2D(xi1,xi2));
            N3 = N_3(cartCoord2D(xi1,xi2));
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            g = pGpn2(k,normal,x,y);
            temp[0] = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd;
            temp[1] = temp[0]*N1;
            temp[2] = temp[0]*N2;
            temp[3] = temp[0]*N3;
            *hCoeff1 = cuCaddf(*hCoeff1,make_cuFloatComplex(temp[1]*cuCrealf(g),
                    temp[1]*cuCimagf(g)));
            *hCoeff2 = cuCaddf(*hCoeff2,make_cuFloatComplex(temp[2]*cuCrealf(g),
                    temp[2]*cuCimagf(g)));
            *hCoeff3 = cuCaddf(*hCoeff3,make_cuFloatComplex(temp[3]*cuCrealf(g),
                    temp[3]*cuCimagf(g)));
        }
    }
}

__device__ void h_l_sgl2(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *hCoeff1, 
        cuFloatComplex *hCoeff2, cuFloatComplex *hCoeff3) {
    *hCoeff1 = make_cuFloatComplex(0,0);
    *hCoeff2 = make_cuFloatComplex(0,0);
    *hCoeff3 = make_cuFloatComplex(0,0);
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, N1, N2, N3, temp[4];
    cartCoord y, crossProd, normal;
    cuFloatComplex g;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    normal = crossProd.nrmlzd();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = rho*theta;
            xi2 = 1-rho;
            N1 = N_1(cartCoord2D(xi1,xi2));
            N2 = N_2(cartCoord2D(xi1,xi2));
            N3 = N_3(cartCoord2D(xi1,xi2));
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            g = pGpn2(k,normal,x,y);
            temp[0] = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd;
            temp[1] = temp[0]*N1;
            temp[2] = temp[0]*N2;
            temp[3] = temp[0]*N3;
            *hCoeff1 = cuCaddf(*hCoeff1,make_cuFloatComplex(temp[1]*cuCrealf(g),
                    temp[1]*cuCimagf(g)));
            *hCoeff2 = cuCaddf(*hCoeff2,make_cuFloatComplex(temp[2]*cuCrealf(g),
                    temp[2]*cuCimagf(g)));
            *hCoeff3 = cuCaddf(*hCoeff3,make_cuFloatComplex(temp[3]*cuCrealf(g),
                    temp[3]*cuCimagf(g)));
        }
    }
}

__device__ void h_l_sgl3(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, cuFloatComplex *hCoeff1, 
        cuFloatComplex *hCoeff2, cuFloatComplex *hCoeff3) {
    *hCoeff1 = make_cuFloatComplex(0,0);
    *hCoeff2 = make_cuFloatComplex(0,0);
    *hCoeff3 = make_cuFloatComplex(0,0);
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, N1, N2, N3, temp[4];
    cartCoord y, crossProd, normal;
    cuFloatComplex g;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    normal = crossProd.nrmlzd();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = rho*(1-theta);
            xi2 = rho*theta;
            N1 = N_1(cartCoord2D(xi1,xi2));
            N2 = N_2(cartCoord2D(xi1,xi2));
            N3 = N_3(cartCoord2D(xi1,xi2));
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            g = pGpn2(k,normal,x,y);
            temp[0] = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd;
            temp[1] = temp[0]*N1;
            temp[2] = temp[0]*N2;
            temp[3] = temp[0]*N3;
            *hCoeff1 = cuCaddf(*hCoeff1,make_cuFloatComplex(temp[1]*cuCrealf(g),
                    temp[1]*cuCimagf(g)));
            *hCoeff2 = cuCaddf(*hCoeff2,make_cuFloatComplex(temp[2]*cuCrealf(g),
                    temp[2]*cuCimagf(g)));
            *hCoeff3 = cuCaddf(*hCoeff3,make_cuFloatComplex(temp[3]*cuCrealf(g),
                    temp[3]*cuCimagf(g)));
        }
    }
}

__device__ void c_l_nsgl(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, float *cCoeff) {
    *cCoeff = 0;
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, temp;
    cartCoord y, crossProd, normal;
    float psi;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    normal = crossProd.nrmlzd();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = rho*(1-theta);
            xi2 = rho*theta;
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            psi = pPsiLpn2(normal,x,y);
            temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd;
            *cCoeff += temp*psi;
        }
    }
}

__device__ void c_l_sgl1(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, float *cCoeff) {
    *cCoeff = 0;
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, temp;
    cartCoord y, crossProd, normal;
    float psi;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    normal = crossProd.nrmlzd();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = 1-rho;
            xi2 = rho*(1-theta);
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            psi = pPsiLpn2(normal,x,y);
            temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd;
            *cCoeff += temp*psi;
        }
    }
}

__device__ void c_l_sgl2(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, float *cCoeff) {
    *cCoeff = 0;
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, temp;
    cartCoord y, crossProd, normal;
    float psi;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    normal = crossProd.nrmlzd();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = rho*theta;
            xi2 = 1-rho;
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            psi = pPsiLpn2(normal,x,y);
            temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd;
            *cCoeff += temp*psi;
        }
    }
}

__device__ void c_l_sgl3(const float k, const cartCoord x, const cartCoord p1, 
        const cartCoord p2, const cartCoord p3, float *cCoeff) {
    *cCoeff = 0;
    float eta1, eta2, xi1, xi2, rho, theta, vertCrossProd, temp;
    cartCoord y, crossProd, normal;
    float psi;
    crossProd = (p1-p3)*(p2-p3);
    vertCrossProd = crossProd.nrm2();
    normal = crossProd.nrmlzd();
    int n, m;
    for(n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        for(m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            rho = 0.5+0.5*eta1;
            theta = 0.5+0.5*eta2;
            xi1 = rho*(1-theta);
            xi2 = rho*theta;
            y = xiToElem(p1,p2,p3,cartCoord2D(xi1,xi2));
            psi = pPsiLpn2(normal,x,y);
            temp = 0.25*INTWGTS[n]*INTWGTS[m]*rho*vertCrossProd;
            *cCoeff += temp*psi;
        }
    }
}

__device__ void g_h_c_l_nsgl(const float &k, const cartCoord &x, const cartCoord p[3], 
        cuFloatComplex gCoeff[3], cuFloatComplex hCoeff[3], float *cCoeff) {
    //Initalization of g, h and c
    //printf("(%f,%f,%f)\n",p[0].coords[0],p[0].coords[1],p[0].coords[2]);
    for(int i=0;i<3;i++) {
        gCoeff[i] = make_cuFloatComplex(0,0);
        hCoeff[i] = make_cuFloatComplex(0,0);
    }
    *cCoeff = 0;
    
    //Local variables
    float eta1, eta2, wn, wm, xi1, xi2, rho, theta, vertCrossProd, N1, N2, N3, temp, 
            temp_gh[3], omega = k*speed, pPsiLpn2, radius, prpn2;
    cartCoord y, crossProd, normal, rVec;
    cuFloatComplex Psi, pPsipn2;
    crossProd = (p[0]-p[2])*(p[1]-p[2]);
    //printf("(%f,%f,%f)\n",crossProd.coords[0],crossProd.coords[1],crossProd.coords[2]);
    vertCrossProd = sqrtf((crossProd.coords[0]*crossProd.coords[0])
            +(crossProd.coords[1]*crossProd.coords[1])+(crossProd.coords[2]*crossProd.coords[2]));
    //printf("vert: %f\n",vertCrossProd);
    normal = crossProd.nrmlzd();
    //printf("normal=(%f,%f,%f)\n",normal.coords[0],normal.coords[1],normal.coords[2]);
    const float prodRhoOmega = density*omega;
    const float fourPI = 4.0f*PI;
    //printf("density*omega = %f\n",prodRhoOmega);
    //Compute integrals for g, h and c
    for(int n=0;n<INTORDER;n++) {
        eta2 = INTPNTS[n];
        wn = INTWGTS[n];
        theta = 0.5f+0.5f*eta2;
        for(int m=0;m<INTORDER;m++) {
            eta1 = INTPNTS[m];
            wm = INTWGTS[m];
            rho = 0.5f+0.5f*eta1;
            temp = 0.25f*wn*wm*rho*vertCrossProd;
            
            xi1 = rho*(1-theta);
            xi2 = rho-xi1;
            //printf("xi1 = %f, xi2 = %f\n",xi1,xi2);
            y = xiToElem(p[0],p[1],p[2],cartCoord2D(xi1,xi2));
            //printf("x: (%f,%f,%f), y: (%f,%f,%f)\n",x.coords[0],x.coords[1],x.coords[2],
            //        y.coords[0],y.coords[1],y.coords[2]);
            rVec = y-x;
            radius = sqrtf(rVec.coords[0]*rVec.coords[0]+rVec.coords[1]*rVec.coords[1]+rVec.coords[2]*rVec.coords[2]);
            //printf("radius = %f\n",radius);
            prpn2 = ((y.coords[0]-x.coords[0])*normal.coords[0]+(y.coords[1]-x.coords[1])*normal.coords[1]
                    +(y.coords[2]-x.coords[2])*normal.coords[2])/radius;
            //printf("prpn2=%f\n",prpn2);
            pPsiLpn2 = -1.0f/fourPI/(radius*radius)*prpn2;
            //printf("%f\n",pPsiLpn2);
            Psi = make_cuFloatComplex(__cosf(-k*radius)/(fourPI*radius),__sinf(-k*radius)/(fourPI*radius));
            pPsipn2 = cuCmulf(Psi,make_cuFloatComplex(-1.0f/radius,-k));
            pPsipn2 = make_cuFloatComplex(prpn2*cuCrealf(pPsipn2),prpn2*cuCimagf(pPsipn2));
            N1 = xi1;
            N2 = xi2;
            N3 = 1.0f-xi1-xi2;
            temp_gh[0] = temp*N1;
            temp_gh[1] = temp*N2;
            temp_gh[2] = temp*N3;
            
            gCoeff[0] = cuCaddf(gCoeff[0],make_cuFloatComplex(temp_gh[0]*cuCrealf(Psi),temp_gh[0]*cuCimagf(Psi)));
            gCoeff[1] = cuCaddf(gCoeff[1],make_cuFloatComplex(temp_gh[1]*cuCrealf(Psi),temp_gh[1]*cuCimagf(Psi)));
            gCoeff[2] = cuCaddf(gCoeff[2],make_cuFloatComplex(temp_gh[2]*cuCrealf(Psi),temp_gh[2]*cuCimagf(Psi)));
            
            hCoeff[0] = cuCaddf(hCoeff[0],make_cuFloatComplex(temp_gh[0]*cuCrealf(pPsipn2),temp_gh[0]*cuCimagf(pPsipn2)));
            hCoeff[1] = cuCaddf(hCoeff[1],make_cuFloatComplex(temp_gh[1]*cuCrealf(pPsipn2),temp_gh[1]*cuCimagf(pPsipn2)));
            hCoeff[2] = cuCaddf(hCoeff[2],make_cuFloatComplex(temp_gh[2]*cuCrealf(pPsipn2),temp_gh[2]*cuCimagf(pPsipn2)));
            
            *cCoeff += temp*pPsiLpn2;
        }
    }
    gCoeff[0] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff[0]),prodRhoOmega*cuCrealf(gCoeff[0]));
    gCoeff[1] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff[1]),prodRhoOmega*cuCrealf(gCoeff[1]));
    gCoeff[2] = make_cuFloatComplex(-prodRhoOmega*cuCimagf(gCoeff[2]),prodRhoOmega*cuCrealf(gCoeff[2]));
    //printComplexMatrix(gCoeff,1,1,1);
}

//singular not dealt with in this function
__global__ void elemLPnts_nsgl(const float k, const int l, const triElem *elems, const cartCoord *pnts, 
        const int numNods, const int numCHIEF, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int numSrcs, const int ldb) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    if(idx<numNods+numCHIEF && idx!=elems[l].nodes[0] && idx!=elems[l].nodes[1] 
            && idx!=elems[l].nodes[2]) {
        int i, j;
        cuFloatComplex hCoeffs[3], gCoeffs[3], bc, pCoeffs[3];
        float cCoeff;
        h_l_nsgl(k,pnts[idx],pnts[elems[l].nodes[0]],pnts[elems[l].nodes[1]],pnts[elems[l].nodes[2]],
                hCoeffs,hCoeffs+1,hCoeffs+2);
        g_l_nsgl(k,pnts[idx],pnts[elems[l].nodes[0]],pnts[elems[l].nodes[1]],pnts[elems[l].nodes[2]],
                gCoeffs,gCoeffs+1,gCoeffs+2);
        c_l_nsgl(k,pnts[idx],pnts[elems[l].nodes[0]],pnts[elems[l].nodes[1]],pnts[elems[l].nodes[2]],
                &cCoeff); 
        
        //Update the A matrix
        bc = cuCdivf(elems[l].bc[0],elems[l].bc[1]);
        for(i=0;i<3;i++) {
            pCoeffs[i] = cuCsubf(hCoeffs[i],cuCmulf(bc,gCoeffs[i]));
        }
        for(i=0;i<3;i++) {
            A[IDXC0(idx,elems[l].nodes[i],lda)] = cuCaddf(A[IDXC0(idx,elems[l].nodes[i],lda)],pCoeffs[i]);
        }
        
        //Update from C coefficients
        if(idx<numNods) {
            A[IDXC0(idx,idx,lda)] = cuCsubf(A[IDXC0(idx,idx,lda)],make_cuFloatComplex(cCoeff,0));
        }
        
        //Update the B matrix
        bc = cuCdivf(elems[l].bc[2],elems[l].bc[1]);
        for(i=0;i<numSrcs;i++) {
            for(j=0;j<3;j++) {
                B[IDXC0(idx,i,ldb)] = cuCsubf(B[IDXC0(idx,i,ldb)],
                        cuCmulf(bc,gCoeffs[j]));
                
            }
        }
        
    }
}

__global__ void elemsPnts_sgl(const float k, const triElem *elems, const int numElems,
        const cartCoord *pnts, cuFloatComplex *hCoeffs_sgl1, cuFloatComplex *hCoeffs_sgl2, 
        cuFloatComplex *hCoeffs_sgl3, cuFloatComplex *gCoeffs_sgl1, cuFloatComplex *gCoeffs_sgl2, 
        cuFloatComplex *gCoeffs_sgl3, float *cCoeffs_sgl1, float *cCoeffs_sgl2, float *cCoeffs_sgl3) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numElems) {
        triElem elem = elems[idx];
        h_l_sgl1(k,pnts[elem.nodes[0]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
                hCoeffs_sgl1+3*idx,hCoeffs_sgl1+3*idx+1,hCoeffs_sgl1+3*idx+2);
        h_l_sgl2(k,pnts[elem.nodes[1]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
                hCoeffs_sgl2+3*idx,hCoeffs_sgl2+3*idx+1,hCoeffs_sgl2+3*idx+2);
        h_l_sgl3(k,pnts[elem.nodes[2]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
                hCoeffs_sgl3+3*idx,hCoeffs_sgl3+3*idx+1,hCoeffs_sgl3+3*idx+2);
        
        g_l_sgl1(k,pnts[elem.nodes[0]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
                gCoeffs_sgl1+3*idx,gCoeffs_sgl1+3*idx+1,gCoeffs_sgl1+3*idx+2);
        g_l_sgl2(k,pnts[elem.nodes[1]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
                gCoeffs_sgl2+3*idx,gCoeffs_sgl2+3*idx+1,gCoeffs_sgl2+3*idx+2);
        g_l_sgl3(k,pnts[elem.nodes[2]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
                gCoeffs_sgl3+3*idx,gCoeffs_sgl3+3*idx+1,gCoeffs_sgl3+3*idx+2);
        
        c_l_sgl1(k,pnts[elem.nodes[0]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
                cCoeffs_sgl1+idx);
        c_l_sgl2(k,pnts[elem.nodes[1]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
                cCoeffs_sgl2+idx);
        c_l_sgl3(k,pnts[elem.nodes[2]],pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
                cCoeffs_sgl3+idx);
    }
}

__global__ void updateSystem_sgl(const triElem *elems, const int numElems, 
        cuFloatComplex *hCoeffs_sgl1, cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, 
        cuFloatComplex *gCoeffs_sgl1, cuFloatComplex *gCoeffs_sgl2, cuFloatComplex *gCoeffs_sgl3, 
        cuFloatComplex *A, const int lda) {
    //Indices with the same row and column index must be updated on the CPU!
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numElems) {
        int i;
        triElem elem = elems[idx];
        cuFloatComplex pCoeffs[3];
        cuFloatComplex bc = cuCdivf(elem.bc[0],elem.bc[1]);
        
        //Deal with singularity 1
        for(i=0;i<3;i++) {
            pCoeffs[i] = cuCsubf(hCoeffs_sgl1[3*idx+i],cuCmulf(bc,gCoeffs_sgl1[3*idx+i]));
        }
        A[IDXC0(elem.nodes[0],elem.nodes[1],lda)] = cuCaddf(A[IDXC0(elem.nodes[0],elem.nodes[1],lda)],
                pCoeffs[1]);
        A[IDXC0(elem.nodes[0],elem.nodes[2],lda)] = cuCaddf(A[IDXC0(elem.nodes[0],elem.nodes[2],lda)],
                pCoeffs[2]);
        
        //Deal with singularity 2
        for(i=0;i<3;i++) {
            pCoeffs[i] = cuCsubf(hCoeffs_sgl2[3*idx+i],cuCmulf(bc,gCoeffs_sgl2[3*idx+i]));
        }
        A[IDXC0(elem.nodes[1],elem.nodes[0],lda)] = cuCaddf(A[IDXC0(elem.nodes[1],elem.nodes[0],lda)],
                pCoeffs[0]);
        A[IDXC0(elem.nodes[1],elem.nodes[2],lda)] = cuCaddf(A[IDXC0(elem.nodes[1],elem.nodes[2],lda)],
                pCoeffs[2]);
        
        //Deal with singularity 3
        for(i=0;i<3;i++) {
            pCoeffs[i] = cuCsubf(hCoeffs_sgl3[3*idx+i],cuCmulf(bc,gCoeffs_sgl3[3*idx+i]));
        }
        A[IDXC0(elem.nodes[2],elem.nodes[0],lda)] = cuCaddf(A[IDXC0(elem.nodes[2],elem.nodes[0],lda)],
                pCoeffs[0]);
        A[IDXC0(elem.nodes[2],elem.nodes[1],lda)] = cuCaddf(A[IDXC0(elem.nodes[2],elem.nodes[1],lda)],
                pCoeffs[1]);
    }
}

void updateSystemCPU(const triElem *elems, const int numElems, 
        cuFloatComplex *hCoeffs_sgl1, cuFloatComplex *hCoeffs_sgl2, cuFloatComplex *hCoeffs_sgl3, 
        cuFloatComplex *gCoeffs_sgl1, cuFloatComplex *gCoeffs_sgl2, cuFloatComplex *gCoeffs_sgl3, 
        float *cCoeffs_sgl1, float *cCoeffs_sgl2, float *cCoeffs_sgl3,
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrcs, const int ldb) {
    int i, j, k;
    cuFloatComplex bc, pCoeff;
    for(i=0;i<numElems;i++) {
        bc = cuCdivf(elems[i].bc[0],elems[i].bc[1]);
        pCoeff = cuCsubf(hCoeffs_sgl1[3*i],cuCmulf(bc,gCoeffs_sgl1[3*i]));
        A[IDXC0(elems[i].nodes[0],elems[i].nodes[0],lda)] = cuCaddf(A[IDXC0(elems[i].nodes[0],elems[i].nodes[0],lda)],
            pCoeff);
        pCoeff = cuCsubf(hCoeffs_sgl2[3*i+1],cuCmulf(bc,gCoeffs_sgl2[3*i+1]));
        A[IDXC0(elems[i].nodes[1],elems[i].nodes[1],lda)] = cuCaddf(A[IDXC0(elems[i].nodes[1],elems[i].nodes[1],lda)],
            pCoeff);
        pCoeff = cuCsubf(hCoeffs_sgl3[3*i+2],cuCmulf(bc,gCoeffs_sgl3[3*i+2]));
        A[IDXC0(elems[i].nodes[2],elems[i].nodes[2],lda)] = cuCaddf(A[IDXC0(elems[i].nodes[2],elems[i].nodes[2],lda)],
            pCoeff);
        
        A[IDXC0(elems[i].nodes[0],elems[i].nodes[0],lda)] = cuCsubf(A[IDXC0(elems[i].nodes[0],elems[i].nodes[0],lda)],
                make_cuFloatComplex(cCoeffs_sgl1[i],0));
        A[IDXC0(elems[i].nodes[1],elems[i].nodes[1],lda)] = cuCsubf(A[IDXC0(elems[i].nodes[1],elems[i].nodes[1],lda)],
                make_cuFloatComplex(cCoeffs_sgl2[i],0));
        A[IDXC0(elems[i].nodes[2],elems[i].nodes[2],lda)] = cuCsubf(A[IDXC0(elems[i].nodes[2],elems[i].nodes[2],lda)],
                make_cuFloatComplex(cCoeffs_sgl3[i],0));
        
        
        bc = cuCdivf(elems[i].bc[2],elems[i].bc[1]);
        //std::cout << "boundary condition: " << bc << std::endl;
        for(j=0;j<numSrcs;j++) {
            for(k=0;k<3;k++) {
                B[IDXC0(elems[i].nodes[0],j,ldb)] = cuCsubf(B[IDXC0(elems[i].nodes[0],j,ldb)],
                    cuCmulf(bc,gCoeffs_sgl1[3*i+k]));
                B[IDXC0(elems[i].nodes[1],j,ldb)] = cuCsubf(B[IDXC0(elems[i].nodes[1],j,ldb)],
                    cuCmulf(bc,gCoeffs_sgl2[3*i+k]));
                B[IDXC0(elems[i].nodes[2],j,ldb)] = cuCsubf(B[IDXC0(elems[i].nodes[2],j,ldb)],
                    cuCmulf(bc,gCoeffs_sgl3[3*i+k]));
            }
        }
    }
}

int genSystem(const float k, const triElem *elems, const int numElems, 
        const cartCoord *pnts, const int numNods, const int numCHIEF, 
        const cartCoord *srcs, const int numSrcs, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int ldb) {
    //Initialization of A
    int i, j, l;
    for(i=0;i<numNods+numCHIEF;i++) {
        for(j=0;j<numNods;j++) {
            if(i==j) {
                A[IDXC0(i,j,lda)] = make_cuFloatComplex(1,0);
            } else {
                A[IDXC0(i,j,lda)] = make_cuFloatComplex(0,0);
            }
        }
    }
    
    //Initialization of B
    for(i=0;i<numNods+numCHIEF;i++) {
        for(j=0;j<numSrcs;j++) {
            B[IDXC0(i,j,ldb)] = pntSrc(k,STRENGTH,srcs[j],pnts[i]);
        }
    }
    
    //Move elements to GPU
    triElem *elems_d;
    CUDA_CALL(cudaMalloc(&elems_d,numElems*sizeof(triElem)));
    CUDA_CALL(cudaMemcpy(elems_d,elems,numElems*sizeof(triElem),cudaMemcpyHostToDevice));
    
    //Move points to GPU
    cartCoord *pnts_d;
    CUDA_CALL(cudaMalloc(&pnts_d,(numNods+numCHIEF)*sizeof(cartCoord)));
    CUDA_CALL(cudaMemcpy(pnts_d,pnts,(numNods+numCHIEF)*sizeof(cartCoord),cudaMemcpyHostToDevice));
    
    
    cuFloatComplex *A_d, *B_d;
    CUDA_CALL(cudaMalloc(&A_d,(numNods+numCHIEF)*numNods*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(A_d,A,(numNods+numCHIEF)*numNods*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&B_d,(numNods+numCHIEF)*numSrcs*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMemcpy(B_d,B,(numNods+numCHIEF)*numSrcs*sizeof(cuFloatComplex),cudaMemcpyHostToDevice));
    
    int numBlocks, width = 32;
    numBlocks = (numNods+numCHIEF+width-1)/width;
    //std::cout << "Number of blocks: " << numBlocks << std::endl;
    for(l=0;l<numElems;l++) {
        elemLPnts_nsgl<<<numBlocks,width>>>(k,l,elems_d,pnts_d,numNods,numCHIEF,
                A_d,lda,B_d,numSrcs,ldb);
    }
    
    
    //std::cout << "completed kernel." << std::endl;
    //Update singular
    cuFloatComplex *hCoeffs_sgl1, *hCoeffs_sgl2, *hCoeffs_sgl3, *gCoeffs_sgl1, 
            *gCoeffs_sgl2, *gCoeffs_sgl3, *hCoeffs_sgl1_d, *hCoeffs_sgl2_d, 
            *hCoeffs_sgl3_d, *gCoeffs_sgl1_d, *gCoeffs_sgl2_d, *gCoeffs_sgl3_d;
    float *cCoeffs_sgl1, *cCoeffs_sgl2, *cCoeffs_sgl3, 
            *cCoeffs_sgl1_d, *cCoeffs_sgl2_d, *cCoeffs_sgl3_d;
    
    hCoeffs_sgl1 = new cuFloatComplex[3*numElems];
    hCoeffs_sgl2 = new cuFloatComplex[3*numElems];
    hCoeffs_sgl3 = new cuFloatComplex[3*numElems];
    gCoeffs_sgl1 = new cuFloatComplex[3*numElems];
    gCoeffs_sgl2 = new cuFloatComplex[3*numElems];
    gCoeffs_sgl3 = new cuFloatComplex[3*numElems];
    
    cCoeffs_sgl1 = new float[numElems];
    cCoeffs_sgl2 = new float[numElems];
    cCoeffs_sgl3 = new float[numElems];
    
    CUDA_CALL(cudaMalloc(&hCoeffs_sgl1_d,3*numElems*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMalloc(&hCoeffs_sgl2_d,3*numElems*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMalloc(&hCoeffs_sgl3_d,3*numElems*sizeof(cuFloatComplex)));
    
    CUDA_CALL(cudaMalloc(&gCoeffs_sgl1_d,3*numElems*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMalloc(&gCoeffs_sgl2_d,3*numElems*sizeof(cuFloatComplex)));
    CUDA_CALL(cudaMalloc(&gCoeffs_sgl3_d,3*numElems*sizeof(cuFloatComplex)));
    
    CUDA_CALL(cudaMalloc(&cCoeffs_sgl1_d,numElems*sizeof(float)));
    CUDA_CALL(cudaMalloc(&cCoeffs_sgl2_d,numElems*sizeof(float)));
    CUDA_CALL(cudaMalloc(&cCoeffs_sgl3_d,numElems*sizeof(float)));
    
    numBlocks = (numElems+width-1)/width;
    elemsPnts_sgl<<<numBlocks,width>>>(k,elems_d,numElems,pnts_d,hCoeffs_sgl1_d,hCoeffs_sgl2_d,hCoeffs_sgl3_d,
            gCoeffs_sgl1_d,gCoeffs_sgl2_d,gCoeffs_sgl3_d,cCoeffs_sgl1_d,cCoeffs_sgl2_d,
            cCoeffs_sgl3_d);
    
    updateSystem_sgl<<<numBlocks,width>>>(elems_d,numElems,hCoeffs_sgl1_d,hCoeffs_sgl2_d,hCoeffs_sgl3_d,
            gCoeffs_sgl1_d,gCoeffs_sgl2_d,gCoeffs_sgl3_d,A_d,lda);
    
    CUDA_CALL(cudaMemcpy(hCoeffs_sgl1,hCoeffs_sgl1_d,3*numElems*sizeof(cuFloatComplex),
            cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hCoeffs_sgl2,hCoeffs_sgl2_d,3*numElems*sizeof(cuFloatComplex),
            cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hCoeffs_sgl3,hCoeffs_sgl3_d,3*numElems*sizeof(cuFloatComplex),
            cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaMemcpy(gCoeffs_sgl1,gCoeffs_sgl1_d,3*numElems*sizeof(cuFloatComplex),
            cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(gCoeffs_sgl2,gCoeffs_sgl2_d,3*numElems*sizeof(cuFloatComplex),
            cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(gCoeffs_sgl3,gCoeffs_sgl3_d,3*numElems*sizeof(cuFloatComplex),
            cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaMemcpy(cCoeffs_sgl1,cCoeffs_sgl1_d,numElems*sizeof(float),
            cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(cCoeffs_sgl2,cCoeffs_sgl2_d,numElems*sizeof(float),
            cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(cCoeffs_sgl3,cCoeffs_sgl3_d,numElems*sizeof(float),
            cudaMemcpyDeviceToHost));
    
    
    
    CUDA_CALL(cudaMemcpy(A,A_d,(numNods+numCHIEF)*numNods*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(B,B_d,(numNods+numCHIEF)*numSrcs*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    //std::cout << "B: " << std::endl;
    //printComplexMatrix(B,numNods+numCHIEF,numSrcs,numNods+numCHIEF);
    
    
    updateSystemCPU(elems,numElems,hCoeffs_sgl1,hCoeffs_sgl2,hCoeffs_sgl3,
            gCoeffs_sgl1,gCoeffs_sgl2,gCoeffs_sgl3,cCoeffs_sgl1,cCoeffs_sgl2,cCoeffs_sgl3,
            A,lda,B,numSrcs,ldb);
    
    
    CUDA_CALL(cudaFree(A_d));
    CUDA_CALL(cudaFree(B_d));
    CUDA_CALL(cudaFree(hCoeffs_sgl1_d));
    CUDA_CALL(cudaFree(hCoeffs_sgl2_d));
    CUDA_CALL(cudaFree(hCoeffs_sgl3_d));
    CUDA_CALL(cudaFree(gCoeffs_sgl1_d));
    CUDA_CALL(cudaFree(gCoeffs_sgl2_d));
    CUDA_CALL(cudaFree(gCoeffs_sgl3_d));
    CUDA_CALL(cudaFree(cCoeffs_sgl1_d));
    CUDA_CALL(cudaFree(cCoeffs_sgl2_d));
    CUDA_CALL(cudaFree(cCoeffs_sgl3_d));
    CUDA_CALL(cudaFree(elems_d));
    CUDA_CALL(cudaFree(pnts_d));
    
    delete[] hCoeffs_sgl1;
    delete[] hCoeffs_sgl2;
    delete[] hCoeffs_sgl3;
    
    delete[] gCoeffs_sgl1;
    delete[] gCoeffs_sgl2;
    delete[] gCoeffs_sgl3;
    
    delete[] cCoeffs_sgl1;
    delete[] cCoeffs_sgl2;
    delete[] cCoeffs_sgl3;
    
    
    return EXIT_SUCCESS;
}

int bemSystem(const mesh &m, const float k, const cartCoord *srcs, const int numSrcs,
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int ldb) {
    int i;
    cartCoord *pnts_d;
    triElem *elems_d;
    std::cout << "Number of nodes: " << m.numPnts << ", number of CHIEF: " << m.numCHIEF 
            << std::endl;
    HOST_CALL(m.meshToGPU(&pnts_d,&elems_d));
    cartCoord *pnts = new cartCoord[m.numPnts+m.numCHIEF];
    for(i=0;i<m.numPnts;i++) {
        pnts[i] = m.pnts[i];
    }
    for(i=0;i<m.numCHIEF;i++) {
        pnts[m.numPnts+i] = m.chiefPnts[i];
    }
    HOST_CALL(atomicGenSystem(k,m.elems,m.numElems,pnts,m.numPnts,m.numCHIEF,srcs,
            numSrcs,A,lda,B,ldb));
    delete[] pnts;
    CUDA_CALL(cudaFree(pnts_d));
    CUDA_CALL(cudaFree(elems_d));
    return EXIT_SUCCESS;
}

__device__ cuFloatComplex pntElemOffset(const float k, const cartCoord x, 
        const triElem elem, const cartCoord *pnts, const cuFloatComplex *surfPressure) {
    int i;
    cuFloatComplex contrib = make_cuFloatComplex(0,0);
    cuFloatComplex hCoeffs[3], gCoeffs[3], pCoeffs[3], bc;
    g_l_nsgl(k,x,pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
            gCoeffs,gCoeffs+1,gCoeffs+2);
    h_l_nsgl(k,x,pnts[elem.nodes[0]],pnts[elem.nodes[1]],pnts[elem.nodes[2]],
            hCoeffs,hCoeffs+1,hCoeffs+2);
    bc = cuCdivf(elem.bc[0],elem.bc[1]);
    for(i=0;i<3;i++) {
        pCoeffs[i] = cuCsubf(hCoeffs[i],cuCmulf(bc,gCoeffs[i]));
        contrib = cuCaddf(contrib,cuCmulf(pCoeffs[i],surfPressure[elem.nodes[i]]));
    }
    bc = cuCdivf(elem.bc[2],elem.bc[1]);
    for(i=0;i<3;i++) {
        contrib = cuCaddf(contrib,cuCmulf(bc,gCoeffs[i]));
    }
    return contrib;
}

__global__ void pntElemsOffset(const float k, const cartCoord x, const triElem *elems, 
        const int numElems, const cartCoord *pnts, const cuFloatComplex *surfPressure, 
        cuFloatComplex *contribs) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numElems) {
        contribs[idx] = pntElemOffset(k,x,elems[idx],pnts,surfPressure);
    }
}

cuFloatComplex genExtPressure(const float k, const mesh &m, const cartCoord src,
        const cartCoord x, const cuFloatComplex *surfPressure) {
    cuFloatComplex *surfPressure_d;
    cudaMalloc(&surfPressure_d,m.getNumPnts()*sizeof(cuFloatComplex));
    cudaMemcpy(surfPressure_d,surfPressure,m.getNumPnts()*sizeof(cuFloatComplex),
            cudaMemcpyHostToDevice);
    
    cartCoord *pnts_d;
    triElem *elems_d;
    m.meshCloudToGPU(&pnts_d,&elems_d);
    
    cuFloatComplex *contribs = new cuFloatComplex[m.getNumElems()], *contribs_d;
    cudaMalloc(&contribs_d,m.getNumElems()*sizeof(triElem));
    
    int numBlocks, width = 32;
    numBlocks = (m.getNumElems()+width-1)/width;
    
    pntElemsOffset<<<numBlocks,width>>>(k,x,elems_d,m.getNumElems(),pnts_d,surfPressure_d,
            contribs_d);
    cudaMemcpy(contribs,contribs_d,m.getNumElems()*sizeof(cuFloatComplex),
            cudaMemcpyDeviceToHost);
    
    cuFloatComplex extPressure = pntSrc(k,STRENGTH,src,x);
    int i;
    for(i=0;i<m.getNumElems();i++) {
        extPressure = cuCsubf(extPressure,contribs[i]);
    }
    
    cudaFree(contribs_d);
    cudaFree(elems_d);
    cudaFree(pnts_d);
    cudaFree(surfPressure_d);
    
    free(contribs);
    return extPressure;
}

__host__ __device__ float trnglArea(const cartCoord p1, const cartCoord p2, 
        const cartCoord p3) {
    cartCoord v1 = p1-p3, v2 = p2-p3;
    cartCoord temp = v1*v2;
    return temp.nrm2()/2.0;
}

__global__ void areaTrngls(const triElem *elems, const int numElems, const cartCoord *nods, 
        float *area) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<numElems) {
        area[idx] = trnglArea(nods[elems[idx].nodes[0]],nods[elems[idx].nodes[1]],
                nods[elems[idx].nodes[2]]);
    }
}

float cmptSurfArea(const mesh &m) {
    triElem *elems_d;
    cudaMalloc(&elems_d,m.getNumElems()*sizeof(triElem));
    cudaMemcpy(elems_d,m.elems,m.getNumElems()*sizeof(triElem),cudaMemcpyHostToDevice);
    
    cartCoord *nods_d;
    cudaMalloc(&nods_d,m.getNumPnts()*sizeof(cartCoord));
    cudaMemcpy(nods_d,m.pnts,m.getNumPnts()*sizeof(cartCoord),cudaMemcpyHostToDevice);
    
    float *area_d, *area = new float[m.getNumElems()];
    cudaMalloc(&area_d,m.getNumElems()*sizeof(float));
    int numBlocks, width = 32;
    numBlocks = (m.getNumElems()+width-1)/width;
    areaTrngls<<<numBlocks,width>>>(elems_d,m.getNumElems(),nods_d,area_d);
    cudaMemcpy(area,area_d,m.getNumElems()*sizeof(float),cudaMemcpyDeviceToHost);
    float totalArea = 0;
    for(int i=0;i<m.getNumElems();i++) {
        totalArea += area[i];
    }
    
    cudaFree(area_d);
    cudaFree(nods_d);
    cudaFree(elems_d);
    
    free(area);
    return totalArea;
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

std::ostream& operator<<(std::ostream &out, const triElem &rhs) {
    out << "nodes indices: " << rhs.nodes[0] << ", " << rhs.nodes[1] << ", " 
            << rhs.nodes[2] << " boundary condition: " << "A = " << rhs.bc[0] 
            << ", B = " << rhs.bc[1] << ", C = " << rhs.bc[2];
    return out;
}

//class mesh
int mesh::readObj(const char *file) {
    int temp[3];
    std::vector<cartCoord> pntVec; 
    std::vector<triElem> elemVec;
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
            elem.bc[2] = make_cuFloatComplex(-1,0); // cc=0
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
        numPnts = rhs.numPnts;
        pnts = new cartCoord[numPnts];
        for(int i=0;i<numPnts;i++) {
            pnts[i] = rhs.pnts[i];
        }
    }
    if(rhs.numCHIEF > 0) {
        numCHIEF = rhs.numCHIEF;
        chiefPnts = new cartCoord[numCHIEF];
        for(int i=0;i<numCHIEF;i++) {
            chiefPnts[i] = rhs.chiefPnts[i];
        }
    }
    
    if(rhs.numElems > 0) {
        numElems = rhs.numElems;
        elems = new triElem[numElems];
        for(int i=0;i<numPnts;i++) {
            elems[i] = rhs.elems[i];
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
    
    if(rhs.numCHIEF > 0) {
        if(chiefPnts != NULL) {
            delete[] chiefPnts;
        }
        numCHIEF = rhs.numCHIEF;
        chiefPnts = new cartCoord[numPnts];
        for(int i=0;i<numCHIEF;i++) {
            chiefPnts[i] = rhs.chiefPnts[i];
        }
    } else {
        if(chiefPnts != NULL) {
            delete[] chiefPnts;
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
    std::cout << "Lower x: " << xl << "Higher x: " << xu << "Lower y: " << yl 
            << "Higher y: " << yu << "Lower z: " << zl << "Higher z: " << zu << std::endl;
}

std::ostream& operator<<(std::ostream &out, const mesh &rhs) {
    for(int i=0;i<rhs.numPnts;i++) {
        std::cout << rhs.pnts[i] << std::endl;
    }
    
    for(int i=0;i<rhs.numElems;i++) {
        std::cout << rhs.elems[i] << std::endl;
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
        std::cout << "Not enough mesh information!" << std::endl;
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
    long long seed = 0;
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
            numBlocks = (numPnts+width-1)/width;
            distPntPnts<<<numBlocks,width>>>(sp,pnts_d,numPnts,dists_d);
            CUDA_CALL(cudaMemcpy(dists,dists_d,numPnts*sizeof(float),cudaMemcpyDeviceToHost));
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
    
    return EXIT_SUCCESS;
}

void mesh::printCHIEF() {
    for(int i=0;i<numCHIEF;i++) {
        std::cout << chiefPnts[i] << std::endl;
    }
}

int mesh::meshCloudToGPU(cartCoord **pPnts_d,triElem **pElems_d) const {
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

int mesh::meshToGPU(cartCoord **pPnts_d, triElem **pElems_d) const {
    if(numPnts==0 || numElems==0) {
        std::cout << "The mesh object is incomplete." << std::endl;
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
        CUDA_CALL(cudaMemcpy(*pPnts_d,pnts_h,(numPnts+numCHIEF)*sizeof(cartCoord),
                cudaMemcpyHostToDevice));
        
        CUDA_CALL(cudaMalloc(pElems_d,numElems*sizeof(triElem)));
        CUDA_CALL(cudaMemcpy(*pElems_d,elems,numElems*sizeof(triElem),cudaMemcpyHostToDevice));
        return EXIT_SUCCESS;
    }
}

//nodElems
nodElems::nodElems(const nodElems &rhs) {
    nodNum = rhs.nodNum;
    numElems = rhs.numElems;
    elems = new triElem[numElems];
    for(int i=0;i<numElems;i++) {
        elems[i] = rhs.elems[i];
    }
}

nodElems& nodElems::operator=(const nodElems &rhs) {
    nodNum = rhs.nodNum;
    numElems = rhs.numElems;
    if(elems!=NULL) {
        delete[] elems;
    }
    if(numElems != 0) {
        elems = new triElem[numElems];
        for(int i=0;i<numElems;i++) {
            elems[i] = rhs.elems[i];
        }
    }
    return *this;
}

void nodElems::findElems(const mesh &m, const int nod) {
    int i, j;
    triElem elem;
    bool *flags = new bool[m.getNumElems()];
    for(i=0;i<m.getNumElems();i++) {
        flags[i] = false;
    }
    
    //Find the elements that contribute to node 
    numElems = 0;
    for(i=0;i<m.getNumElems();i++) {
        if(m.elems[i].nodes[0]==nod||m.elems[i].nodes[1]==nod||m.elems[i].nodes[2]==nod) {
            flags[i] = true;
            numElems++;
        }
    }
    
    //create the elems array
    if(elems!=NULL) {
        delete[] elems;
    }
    elems = new triElem[numElems];
    j = 0;
    for(i=0;i<m.getNumElems();i++) {
        if(flags[i]) {
            if(m.elems[i].nodes[0]!=nod) {
                elem.bc[0] = m.elems[i].bc[0];
                elem.bc[1] = m.elems[i].bc[1];
                elem.bc[2] = m.elems[i].bc[2];
                if(m.elems[i].nodes[1]==nod) {
                    //nod is the second node on the element
                    elem.nodes[0] = m.elems[i].nodes[1];
                    elem.nodes[1] = m.elems[i].nodes[2];
                    elem.nodes[2] = m.elems[i].nodes[0];
                } else {
                    //nod is the third node on the element
                    elem.nodes[0] = m.elems[i].nodes[2];
                    elem.nodes[1] = m.elems[i].nodes[0];
                    elem.nodes[2] = m.elems[i].nodes[1];
                }
            } else {
                elem = m.elems[i];
            }
            elems[j] = m.elems[i];
            j++;
        }
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

