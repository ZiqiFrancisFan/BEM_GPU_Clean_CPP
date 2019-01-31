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



using namespace std;

class cartCoord2D;

//class cartCoord
class cartCoord {
    friend ostream& operator<<(ostream&,const cartCoord&);
    friend __host__ __device__ cartCoord pntNumDvd(const cartCoord&,const float);
    friend __host__ __device__ cartCoord numPntMul(const float,const cartCoord&);
    friend __host__ __device__ float dotProd(const cartCoord&,const cartCoord&);
    friend __host__ __device__ cuFloatComplex green2(const float,const cartCoord,const cartCoord);
    friend __host__ __device__ float Psi_L(const cartCoord);
    friend __host__ __device__ cartCoord tf2DTo3D(const cartCoord,const cartCoord,
        const cartCoord,const cartCoord2D);
    friend __host__ __device__ float trnglArea(const cartCoord,const cartCoord);
    
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
    __host__ __device__ cartCoord operator*(const cartCoord&) const;
    __host__ __device__ void print() {printf("(%f,%f,%f)\n",coords[0],coords[1],coords[2]);}
    __host__ __device__ float nrm2() const;
};

ostream& operator<<(ostream&,const cartCoord&);

__host__ __device__ cartCoord pntNumDvd(const cartCoord&,const float);

__host__ __device__ cartCoord numPntMul(const float,const cartCoord&);

__host__ __device__ float dotProd(const cartCoord&,const cartCoord&);

__host__ __device__ cuFloatComplex green(const cartCoord&,const cartCoord&);

__host__ __device__ float Psi_L(const cartCoord);

__host__ __device__ float trnglArea(const cartCoord,const cartCoord);

//class triElem
class triElem {
    friend ostream& operator<<(ostream&,const triElem&);
    friend class mesh;
private:
    int nodes[3];
    cuFloatComplex bc[3];
    
public:
    triElem() = default;
    triElem(const triElem&);
    ~triElem() = default;
    triElem& operator=(const triElem&);
};

ostream& operator<<(ostream&,const triElem&);

//class mesh
class mesh {
    friend ostream& operator<<(ostream&,const mesh&);
private:
    cartCoord *pnts = NULL;
    triElem *elems = NULL;
    int numPnts = 0;
    int numElems = 0;
    
public:
    mesh() = default;
    mesh(const mesh&);
    ~mesh();
    int readObj(const char*);
    mesh& operator=(const mesh&);
};

ostream& operator<<(ostream&,const mesh&);

class cartCoord2D {
    friend __host__ __device__ cartCoord2D pntNumDvd(const cartCoord2D&,const float);
    friend __host__ __device__ cartCoord2D numPntMul(const float,const cartCoord2D&);
    friend __host__ __device__ float N_1(const cartCoord2D);
    friend __host__ __device__ float N_2(const cartCoord2D);
    friend __host__ __device__ float N_3(const cartCoord2D);
    friend __host__ __device__ float pN1pXi1(const cartCoord2D);
    friend __host__ __device__ float pN1pXi2(const cartCoord2D);
    friend __host__ __device__ float pN2pXi1(const cartCoord2D);
    friend __host__ __device__ float pN2pXi2(const cartCoord2D);
    friend __host__ __device__ float pN3pXi1(const cartCoord2D);
    friend __host__ __device__ float pN3pXi2(const cartCoord2D);
    friend __host__ __device__ cartCoord tf2DTo3D(const cartCoord,const cartCoord,
        const cartCoord,const cartCoord2D);
    
private:
    float coords[2];
    
public:
    __host__ __device__ cartCoord2D() {coords[0]=0;coords[1]=0;}
    __host__ __device__ cartCoord2D(const cartCoord2D&);
    __host__ __device__ cartCoord2D(const float x,const float y) {coords[0]=x;coords[1]=y;}
    ~cartCoord2D() = default;
    __host__ __device__ cartCoord2D& operator=(const cartCoord2D&);
    __host__ __device__ void set(const float,const float);
    __host__ __device__ cartCoord2D operator+(const cartCoord2D&) const;
    __host__ __device__ cartCoord2D operator-(const cartCoord2D&) const;
    __host__ __device__ void print() {printf("(%f,%f)\n",coords[0],coords[1]);}
};

__host__ __device__ cartCoord2D pntNumDvd(const cartCoord2D&,const float);

__host__ __device__ cartCoord2D numPntMul(const float,const cartCoord2D&);

__host__ __device__ float N_1(const cartCoord2D);

__host__ __device__ float N_2(const cartCoord2D);

__host__ __device__ float N_3(const cartCoord2D);

__host__ __device__ float pN1pXi1(const cartCoord2D);

__host__ __device__ float pN1pXi2(const cartCoord2D);

__host__ __device__ float pN2pXi1(const cartCoord2D);

__host__ __device__ float pN2pXi2(const cartCoord2D);

__host__ __device__ float pN3pXi1(const cartCoord2D);

__host__ __device__ float pN3pXi2(const cartCoord2D);

__host__ __device__ cartCoord tf2DTo3D(const cartCoord,const cartCoord,
        const cartCoord,const cartCoord2D);



#endif /* MESH_H */

