/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <cuComplex.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
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

__host__ __device__ cuFloatComplex green2(const float k, const cartCoord x, const cartCoord y) {
    float r = (x-y).nrm2();
    return green(k,r);
}

__host__ __device__ float trnglArea(const cartCoord p1, const cartCoord p2) {
    return (p1*p2).nrm2()/2;
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
    if((sp.isEqual(pnt))) {
        temp = sp;
    } else {
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
    }
    printf("(%f,%f,%f)\n",temp.coords[0],temp.coords[1],temp.coords[2]);
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
    if(coords[0]==p.coords[0]&&coords[1]==p.coords[1]&&coords[2]==p.coords[2]) {
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
        if(t1Nrm.isEqual(t2Nrm) && t2Nrm.isEqual(t3Nrm) && t3Nrm.isEqual(t1Nrm)) {
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
triElem::triElem(const triElem &rhs) {
    for(int i=0;i<3;i++) {
        nodes[i] = rhs.nodes[i];
        bc[i] = rhs.bc[i];
    }
}

triElem& triElem::operator=(const triElem &rhs) {
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
    }
}

__global__ void distPntPnts(const cartCoord sp, const cartCoord *pnts, const int numPnts, float *dists) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; 
    if(idx < numPnts) {
        dists[idx] = (pnts[idx]-sp).nrm2();
    }
}

int mesh::genCHIEF(const int num) {
    numCHIEF = num;
    if(chiefPnts != NULL) {
        delete[] chiefPnts;
    }
    chiefPnts = new cartCoord[numCHIEF];
    float threshold = 0.1;
    float randNums[3];
    int width = 32;
    int numBlocks = (numElems+width-1)/width;
    float xRand, yRand, zRand;
    unsigned long long seed = 0;
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGeneratorHost(&gen,CURAND_RNG_PSEUDO_DEFAULT));
    dirCHIEF.set(1.3,3.3,-0.4);
    cartCoord sp;
    cartCoord *pnts_d;
    triElem *elems_d;
    float *dists = new float[numPnts];
    float *dists_d;
    CUDA_CALL(cudaMalloc(&dists_d,numPnts*sizeof(float)));
    HOST_CALL(transToGPU(&pnts_d,&elems_d));
    bool *flags_d;
    bool *flags = new bool[numElems];
    CUDA_CALL(cudaMalloc(&flags_d,numElems*sizeof(bool)));
    int cnt = 0; //counter for number of CHIEF points
    while(cnt < numCHIEF) {
        do {
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed++));
            CURAND_CALL(curandGenerateUniform(gen,randNums,3));
            xRand = descale(xl,xu,randNums[0]);
            yRand = descale(yl,yu,randNums[1]);
            zRand = descale(zl,zu,randNums[2]);
            sp.set(xRand,yRand,zRand);
            rayTrnglsInt<<<numBlocks,width>>>(sp,dirCHIEF,pnts_d,elems_d,numElems,flags_d);
            CUDA_CALL(cudaMemcpy(flags,flags_d,numElems*sizeof(bool),cudaMemcpyDeviceToHost));
            distPntPnts<<<numBlocks,width>>>(sp,pnts,numPnts,dists_d);
            CUDA_CALL(cudaMemcpy(dists,dists_d,numPnts*sizeof(float),cudaMemcpyDeviceToHost));
        } while(!inObj(flags,numElems)||arrayMin(dists,numPnts)>threshold);
        chiefPnts[cnt].set(xRand,yRand,zRand);
        cnt++;
    }
    
    delete[] dists;
    delete[] flags;
    CUDA_CALL(cudaFree(pnts_d));
    CUDA_CALL(cudaFree(elems_d));
    CUDA_CALL(cudaFree(flags_d));
    CUDA_CALL(cudaFree(dists_d));
    for(int i=0;i<numCHIEF;i++) {
        cout << chiefPnts[i] << endl;
    }
    return EXIT_SUCCESS;
}

int mesh::transToGPU(cartCoord **pPnts_d,triElem **pElems_d) {
    if(pnts!=NULL && elems!=NULL) {
        CUDA_CALL(cudaMalloc(pPnts_d,numPnts*sizeof(cartCoord)));
        CUDA_CALL(cudaMemcpy(*pPnts_d,pnts,numPnts*sizeof(cartCoord),cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMalloc(pElems_d,numElems*sizeof(triElem)));
        CUDA_CALL(cudaMemcpy(*pElems_d,elems,numElems*sizeof(triElem),cudaMemcpyHostToDevice));
    }
    return EXIT_SUCCESS;
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

__host__ __device__ cartCoord xiToRv(const cartCoord pnt1, const cartCoord pnt2,
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

