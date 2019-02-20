/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "atomicFuncs.h"
#include "numerical.h"

__global__ void add(float *loc, float *temp, const int num) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < num) {
        atomicAdd(loc,temp[idx]);
    }
}

__global__ void atomicPntsElems_nsgl(const float k, const cartCoord *pnts, const int numNods, 
        const int idxPntStart, const int idxPntEnd, const triElem *elems, const int numElems, 
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrcs, const int ldb) {
    int xIdx = blockIdx.x*blockDim.x+threadIdx.x; //Index for points
    int yIdx = blockIdx.y*blockDim.y+threadIdx.y; //Index for elements
    //The thread with indices xIdx and yIdx process the point xIdx and elem yIdx
    if(xIdx>=idxPntStart && xIdx<=idxPntEnd && yIdx<numElems && xIdx!=elems[yIdx].nodes[0] 
            && xIdx!=elems[yIdx].nodes[1] && xIdx!=elems[yIdx].nodes[2]) {
        int i, j;
        cuFloatComplex hCoeffs[3], gCoeffs[3], bc, pCoeffs[3], temp;
        float cCoeff;
        h_l_nsgl(k,pnts[xIdx],pnts[elems[yIdx].nodes[0]],pnts[elems[yIdx].nodes[1]],pnts[elems[yIdx].nodes[2]],
                hCoeffs,hCoeffs+1,hCoeffs+2);
        
        g_l_nsgl(k,pnts[xIdx],pnts[elems[yIdx].nodes[0]],pnts[elems[yIdx].nodes[1]],pnts[elems[yIdx].nodes[2]],
                gCoeffs,gCoeffs+1,gCoeffs+2);
        
        c_l_nsgl(k,pnts[xIdx],pnts[elems[yIdx].nodes[0]],pnts[elems[yIdx].nodes[1]],pnts[elems[yIdx].nodes[2]],
                &cCoeff);
        
        //Update the A matrix
        bc = cuCdivf(elems[yIdx].bc[0],elems[yIdx].bc[1]);
        for(i=0;i<3;i++) {
            pCoeffs[i] = cuCsubf(hCoeffs[i],cuCmulf(bc,gCoeffs[i]));
        }
        
        for(i=0;i<3;i++) {
            //atomicFloatComplexAdd(&A[IDXC0(xIdx,elems[yIdx].nodes[i],lda)],pCoeffs[i]);
            atomicAdd(&A[IDXC0(xIdx,elems[yIdx].nodes[i],lda)].x,cuCrealf(pCoeffs[i]));
            atomicAdd(&A[IDXC0(xIdx,elems[yIdx].nodes[i],lda)].y,cuCimagf(pCoeffs[i]));
        }
        
        //Update from C coefficients
        if(xIdx<numNods) {
            //atomicFloatComplexSub(&A[IDXC0(xIdx,xIdx,lda)],make_cuFloatComplex(cCoeff,0));
            atomicAdd(&A[IDXC0(xIdx,xIdx,lda)].x,-cCoeff);
        }
        
        //Update the B matrix
        bc = cuCdivf(elems[yIdx].bc[2],elems[yIdx].bc[1]);
        //printf("bc: \n");
        //printComplexMatrix(&bc,1,1,1);
        for(i=0;i<numSrcs;i++) {
            for(j=0;j<3;j++) {
                //atomicFloatComplexSub(&B[IDXC0(xIdx,i,ldb)],cuCmulf(bc,gCoeffs[j]));
                temp = cuCmulf(bc,gCoeffs[j]);
                atomicAdd(&B[IDXC0(xIdx,i,ldb)].x,-cuCrealf(temp));
                atomicAdd(&B[IDXC0(xIdx,i,ldb)].y,-cuCimagf(temp));
            }
        }
    }
}

__global__ void atomicPntsElems_sgl(const float k, const cartCoord *pnts, const triElem *elems, 
        const int numElems, cuFloatComplex *A, const int lda, cuFloatComplex *B, 
        const int numSrcs, const int ldb) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numElems) {
        int i, j;
        cuFloatComplex hCoeffs_sgl1[3], hCoeffs_sgl2[3], hCoeffs_sgl3[3], 
                gCoeffs_sgl1[3], gCoeffs_sgl2[3], gCoeffs_sgl3[3], pCoeffs_sgl1[3], 
                pCoeffs_sgl2[3], pCoeffs_sgl3[3], bc, temp;
        float cCoeff_sgl1, cCoeff_sgl2, cCoeff_sgl3;
        // Compute h and g coefficients
        h_l_sgl1(k,pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],hCoeffs_sgl1,hCoeffs_sgl1+1,hCoeffs_sgl1+2);
        h_l_sgl2(k,pnts[elems[idx].nodes[1]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],hCoeffs_sgl2,hCoeffs_sgl2+1,hCoeffs_sgl2+2);
        h_l_sgl3(k,pnts[elems[idx].nodes[2]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],hCoeffs_sgl3,hCoeffs_sgl3+1,hCoeffs_sgl3+2);
        
        g_l_sgl1(k,pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],gCoeffs_sgl1,gCoeffs_sgl1+1,gCoeffs_sgl1+2);
        g_l_sgl2(k,pnts[elems[idx].nodes[1]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],gCoeffs_sgl2,gCoeffs_sgl2+1,gCoeffs_sgl2+2);
        g_l_sgl3(k,pnts[elems[idx].nodes[2]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],gCoeffs_sgl3,gCoeffs_sgl3+1,gCoeffs_sgl3+2);
        
        c_l_sgl1(k,pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],&cCoeff_sgl1);
        c_l_sgl2(k,pnts[elems[idx].nodes[1]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],&cCoeff_sgl2);
        c_l_sgl3(k,pnts[elems[idx].nodes[2]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],&cCoeff_sgl3);
        
        //Compute p coefficients
        bc = cuCdivf(elems[idx].bc[0],elems[idx].bc[1]);
        for(j=0;j<3;j++) {
            pCoeffs_sgl1[j] = cuCsubf(hCoeffs_sgl1[j],cuCmulf(bc,gCoeffs_sgl1[j]));
            pCoeffs_sgl2[j] = cuCsubf(hCoeffs_sgl2[j],cuCmulf(bc,gCoeffs_sgl2[j]));
            pCoeffs_sgl3[j] = cuCsubf(hCoeffs_sgl3[j],cuCmulf(bc,gCoeffs_sgl3[j]));
        }
        
        //Update matrix A using pCoeffs
        for(j=0;j<3;j++) {
            //atomicFloatComplexAdd(&A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[j],lda)],
            //        pCoeffs_sgl1[j]);
            atomicAdd(&A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[j],lda)].x,
                    cuCrealf(pCoeffs_sgl1[j]));
            atomicAdd(&A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[j],lda)].y,
                    cuCimagf(pCoeffs_sgl1[j]));
            //atomicFloatComplexAdd(&A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[j],lda)],
            //        pCoeffs_sgl2[j]);
            atomicAdd(&A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[j],lda)].x,
                    cuCrealf(pCoeffs_sgl2[j]));
            atomicAdd(&A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[j],lda)].y,
                    cuCimagf(pCoeffs_sgl2[j]));
            //atomicFloatComplexAdd(&A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[j],lda)],
            //        pCoeffs_sgl3[j]);
            atomicAdd(&A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[j],lda)].x,
                    cuCrealf(pCoeffs_sgl3[j]));
            atomicAdd(&A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[j],lda)].y,
                    cuCimagf(pCoeffs_sgl3[j]));
        }
        
        //atomicFloatComplexSub(&A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[0],lda)],
        //        make_cuFloatComplex(cCoeff_sgl1,0));
        atomicAdd(&A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[0],lda)].x,
                -cCoeff_sgl1);
        //atomicFloatComplexSub(&A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[1],lda)],
        //        make_cuFloatComplex(cCoeff_sgl2,0));
        atomicAdd(&A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[1],lda)].x,
                -cCoeff_sgl2);
        //atomicFloatComplexSub(&A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[2],lda)],
        //        make_cuFloatComplex(cCoeff_sgl3,0));
        atomicAdd(&A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[2],lda)].x,
                -cCoeff_sgl3);
        
        //Update matrix B using g Coefficients
        bc = cuCdivf(elems[idx].bc[2],elems[idx].bc[1]);
        for(i=0;i<numSrcs;i++) {
            for(j=0;j<3;j++) {
                //atomicFloatComplexSub(&B[IDXC0(elems[idx].nodes[0],i,ldb)],
                //        cuCmulf(bc,gCoeffs_sgl1[j]));
                temp = cuCmulf(bc,gCoeffs_sgl1[j]);
                atomicAdd(&B[IDXC0(elems[idx].nodes[0],i,ldb)].x,-cuCrealf(temp));
                atomicAdd(&B[IDXC0(elems[idx].nodes[0],i,ldb)].y,-cuCimagf(temp));
                //atomicFloatComplexSub(&B[IDXC0(elems[idx].nodes[1],i,ldb)],
                //        cuCmulf(bc,gCoeffs_sgl2[j]));
                temp = cuCmulf(bc,gCoeffs_sgl2[j]);
                atomicAdd(&B[IDXC0(elems[idx].nodes[1],i,ldb)].x,-cuCrealf(temp));
                atomicAdd(&B[IDXC0(elems[idx].nodes[1],i,ldb)].y,-cuCimagf(temp));
                //atomicFloatComplexSub(&B[IDXC0(elems[idx].nodes[2],i,ldb)],
                //        cuCmulf(bc,gCoeffs_sgl3[j]));
                temp = cuCmulf(bc,gCoeffs_sgl3[j]);
                atomicAdd(&B[IDXC0(elems[idx].nodes[2],i,ldb)].x,-cuCrealf(temp));
                atomicAdd(&B[IDXC0(elems[idx].nodes[2],i,ldb)].y,-cuCimagf(temp));
            }
        }
    }
}

int atomicGenSystem(const float k, const triElem *elems, const int numElems, 
        const cartCoord *pnts, const int numNods, const int numCHIEF, 
        const cartCoord *srcs, const int numSrcs, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int ldb) {
    //Initialization of A
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int i, j;
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
    
    
    
    int xNumBlocks, xWidth = 16, yNumBlocks, yWidth = 16;
    xNumBlocks = (numNods+numCHIEF+xWidth-1)/xWidth;
    yNumBlocks = (numElems+yWidth-1)/yWidth;
    //xNumBlocks = 2;
    //yNumBlocks = 2;
    dim3 gridLayout, blockLayout;
    gridLayout.x = xNumBlocks;
    gridLayout.y = yNumBlocks;
    
    blockLayout.x = xWidth;
    blockLayout.y = yWidth;
    
    //std::cout << "Number of blocks: " << numBlocks << std::endl;
    cudaEventRecord(start);
    atomicPntsElems_nsgl<<<gridLayout,blockLayout>>>(k,pnts_d,numNods,0,numNods+numCHIEF-1,
            elems_d,numElems,A_d,lda,B_d,numSrcs,ldb);
    
    atomicPntsElems_sgl<<<yNumBlocks,yWidth>>>(k,pnts_d,elems_d,numElems,A_d,lda,
            B_d,numSrcs,ldb);
    cudaEventRecord(stop);
    CUDA_CALL(cudaMemcpy(A,A_d,(numNods+numCHIEF)*numNods*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(B,B_d,(numNods+numCHIEF)*numSrcs*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    std::cout << "Elapsed system generation time: " << milliseconds << "milliseconds" << std::endl;
    CUDA_CALL(cudaFree(A_d));
    CUDA_CALL(cudaFree(B_d));
    CUDA_CALL(cudaFree(elems_d));
    CUDA_CALL(cudaFree(pnts_d));
    
    return EXIT_SUCCESS;
}

__global__ void atomicPntsElems_nsgl_test(const float k, const cartCoord *pnts, const int numNods, 
        const int idxPntStart, const int idxPntEnd, const triElem *elems, const int numElems, 
        cuFloatComplex *A, const int lda, cuFloatComplex *B, const int numSrcs, const int ldb) {
    int xIdx = blockIdx.x*blockDim.x+threadIdx.x; //Index for points
    int yIdx = blockIdx.y*blockDim.y+threadIdx.y; //Index for elements
    //The thread with indices xIdx and yIdx process the point xIdx and elem yIdx
    if(xIdx>=idxPntStart && xIdx<=idxPntEnd && yIdx<numElems && xIdx!=elems[yIdx].nodes[0] 
            && xIdx!=elems[yIdx].nodes[1] && xIdx!=elems[yIdx].nodes[2]) {
        int i, j;
        cuFloatComplex hCoeffs[3], gCoeffs[3], bc, pCoeffs[3], temp;
        float cCoeff;
        h_l_nsgl(k,pnts[xIdx],pnts[elems[yIdx].nodes[0]],pnts[elems[yIdx].nodes[1]],pnts[elems[yIdx].nodes[2]],
                hCoeffs,hCoeffs+1,hCoeffs+2);
        
        g_l_nsgl(k,pnts[xIdx],pnts[elems[yIdx].nodes[0]],pnts[elems[yIdx].nodes[1]],pnts[elems[yIdx].nodes[2]],
                gCoeffs,gCoeffs+1,gCoeffs+2);
        
        c_l_nsgl(k,pnts[xIdx],pnts[elems[yIdx].nodes[0]],pnts[elems[yIdx].nodes[1]],pnts[elems[yIdx].nodes[2]],
                &cCoeff);
        
        //Update the A matrix
        bc = cuCdivf(elems[yIdx].bc[0],elems[yIdx].bc[1]);
        for(i=0;i<3;i++) {
            pCoeffs[i] = cuCsubf(hCoeffs[i],cuCmulf(bc,gCoeffs[i]));
        }
        
        for(i=0;i<3;i++) {
            //atomicFloatComplexAdd(&A[IDXC0(xIdx,elems[yIdx].nodes[i],lda)],pCoeffs[i]);
            A[IDXC0(xIdx,elems[yIdx].nodes[i],lda)].x = A[IDXC0(xIdx,elems[yIdx].nodes[i],lda)].x+pCoeffs[i].x;
            A[IDXC0(xIdx,elems[yIdx].nodes[i],lda)].y = A[IDXC0(xIdx,elems[yIdx].nodes[i],lda)].y+pCoeffs[i].y;
            //atomicAdd(&A[IDXC0(xIdx,elems[yIdx].nodes[i],lda)].x,cuCrealf(pCoeffs[i]));
            //atomicAdd(&A[IDXC0(xIdx,elems[yIdx].nodes[i],lda)].y,cuCimagf(pCoeffs[i]));
        }
        
        //Update from C coefficients
        if(xIdx<numNods) {
            //atomicFloatComplexSub(&A[IDXC0(xIdx,xIdx,lda)],make_cuFloatComplex(cCoeff,0));
            A[IDXC0(xIdx,xIdx,lda)].x = A[IDXC0(xIdx,xIdx,lda)].x-cCoeff;
            //atomicAdd(&A[IDXC0(xIdx,xIdx,lda)].x,-cCoeff);
        }
        
        //Update the B matrix
        bc = cuCdivf(elems[yIdx].bc[2],elems[yIdx].bc[1]);
        //printf("bc: \n");
        //printComplexMatrix(&bc,1,1,1);
        for(i=0;i<numSrcs;i++) {
            for(j=0;j<3;j++) {
                //atomicFloatComplexSub(&B[IDXC0(xIdx,i,ldb)],cuCmulf(bc,gCoeffs[j]));
                temp = cuCmulf(bc,gCoeffs[j]);
                B[IDXC0(xIdx,i,ldb)].x = B[IDXC0(xIdx,i,ldb)].x-temp.x;
                B[IDXC0(xIdx,i,ldb)].y = B[IDXC0(xIdx,i,ldb)].y-temp.y;
                //atomicAdd(&B[IDXC0(xIdx,i,ldb)].x,-cuCrealf(temp));
                //atomicAdd(&B[IDXC0(xIdx,i,ldb)].y,-cuCimagf(temp));
            }
        }
    }
}

__global__ void atomicPntsElems_sgl_test(const float k, const cartCoord *pnts, const triElem *elems, 
        const int numElems, cuFloatComplex *A, const int lda, cuFloatComplex *B, 
        const int numSrcs, const int ldb) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < numElems) {
        int i, j;
        cuFloatComplex hCoeffs_sgl1[3], hCoeffs_sgl2[3], hCoeffs_sgl3[3], 
                gCoeffs_sgl1[3], gCoeffs_sgl2[3], gCoeffs_sgl3[3], pCoeffs_sgl1[3], 
                pCoeffs_sgl2[3], pCoeffs_sgl3[3], bc, temp;
        float cCoeff_sgl1, cCoeff_sgl2, cCoeff_sgl3;
        // Compute h and g coefficients
        h_l_sgl1(k,pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],hCoeffs_sgl1,hCoeffs_sgl1+1,hCoeffs_sgl1+2);
        h_l_sgl2(k,pnts[elems[idx].nodes[1]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],hCoeffs_sgl2,hCoeffs_sgl2+1,hCoeffs_sgl2+2);
        h_l_sgl3(k,pnts[elems[idx].nodes[2]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],hCoeffs_sgl3,hCoeffs_sgl3+1,hCoeffs_sgl3+2);
        
        g_l_sgl1(k,pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],gCoeffs_sgl1,gCoeffs_sgl1+1,gCoeffs_sgl1+2);
        g_l_sgl2(k,pnts[elems[idx].nodes[1]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],gCoeffs_sgl2,gCoeffs_sgl2+1,gCoeffs_sgl2+2);
        g_l_sgl3(k,pnts[elems[idx].nodes[2]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],gCoeffs_sgl3,gCoeffs_sgl3+1,gCoeffs_sgl3+2);
        
        c_l_sgl1(k,pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],&cCoeff_sgl1);
        c_l_sgl2(k,pnts[elems[idx].nodes[1]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],&cCoeff_sgl2);
        c_l_sgl3(k,pnts[elems[idx].nodes[2]],pnts[elems[idx].nodes[0]],pnts[elems[idx].nodes[1]],
                pnts[elems[idx].nodes[2]],&cCoeff_sgl3);
        
        //Compute p coefficients
        bc = cuCdivf(elems[idx].bc[0],elems[idx].bc[1]);
        for(j=0;j<3;j++) {
            pCoeffs_sgl1[j] = cuCsubf(hCoeffs_sgl1[j],cuCmulf(bc,gCoeffs_sgl1[j]));
            pCoeffs_sgl2[j] = cuCsubf(hCoeffs_sgl2[j],cuCmulf(bc,gCoeffs_sgl2[j]));
            pCoeffs_sgl3[j] = cuCsubf(hCoeffs_sgl3[j],cuCmulf(bc,gCoeffs_sgl3[j]));
        }
        
        //Update matrix A using pCoeffs
        for(j=0;j<3;j++) {
            //atomicFloatComplexAdd(&A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[j],lda)],
            //        pCoeffs_sgl1[j]);
            A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[j],lda)].x = A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[j],lda)].x
                    +pCoeffs_sgl1[j].x;
            A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[j],lda)].y = A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[j],lda)].y
                    +pCoeffs_sgl1[j].y;
            //atomicAdd(&A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[j],lda)].x,
            //        cuCrealf(pCoeffs_sgl1[j]));
            //atomicAdd(&A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[j],lda)].y,
            //        cuCimagf(pCoeffs_sgl1[j]));
            //atomicFloatComplexAdd(&A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[j],lda)],
            //        pCoeffs_sgl2[j]);
            A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[j],lda)].x = A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[j],lda)].x
                    +pCoeffs_sgl2[j].x;
            A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[j],lda)].y = A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[j],lda)].y
                    +pCoeffs_sgl2[j].y;
            //atomicAdd(&A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[j],lda)].x,
            //        cuCrealf(pCoeffs_sgl2[j]));
            //atomicAdd(&A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[j],lda)].y,
            //        cuCimagf(pCoeffs_sgl2[j]));
            //atomicFloatComplexAdd(&A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[j],lda)],
            //        pCoeffs_sgl3[j]);
            A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[j],lda)].x = A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[j],lda)].x
                    +pCoeffs_sgl3[j].x;
            A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[j],lda)].y = A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[j],lda)].y
                    +pCoeffs_sgl3[j].y;
            //atomicAdd(&A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[j],lda)].x,
            //        cuCrealf(pCoeffs_sgl3[j]));
            //atomicAdd(&A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[j],lda)].y,
            //        cuCimagf(pCoeffs_sgl3[j]));
        }
        
        //atomicFloatComplexSub(&A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[0],lda)],
        //        make_cuFloatComplex(cCoeff_sgl1,0));
        A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[0],lda)].x = A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[0],lda)].x
                -cCoeff_sgl1;
        //atomicAdd(&A[IDXC0(elems[idx].nodes[0],elems[idx].nodes[0],lda)].x,
        //        -cCoeff_sgl1);
        //atomicFloatComplexSub(&A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[1],lda)],
        //        make_cuFloatComplex(cCoeff_sgl2,0));
        A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[1],lda)].x = A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[1],lda)].x
                -cCoeff_sgl2;
        //atomicAdd(&A[IDXC0(elems[idx].nodes[1],elems[idx].nodes[1],lda)].x,
        //        -cCoeff_sgl2);
        //atomicFloatComplexSub(&A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[2],lda)],
        //        make_cuFloatComplex(cCoeff_sgl3,0));
        A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[2],lda)].x = A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[2],lda)].x
                -cCoeff_sgl3;
        //atomicAdd(&A[IDXC0(elems[idx].nodes[2],elems[idx].nodes[2],lda)].x,
        //        -cCoeff_sgl3);
        
        //Update matrix B using g Coefficients
        bc = cuCdivf(elems[idx].bc[2],elems[idx].bc[1]);
        for(i=0;i<numSrcs;i++) {
            for(j=0;j<3;j++) {
                //atomicFloatComplexSub(&B[IDXC0(elems[idx].nodes[0],i,ldb)],
                //        cuCmulf(bc,gCoeffs_sgl1[j]));
                temp = cuCmulf(bc,gCoeffs_sgl1[j]);
                B[IDXC0(elems[idx].nodes[0],i,ldb)].x = B[IDXC0(elems[idx].nodes[0],i,ldb)].x+temp.x;
                B[IDXC0(elems[idx].nodes[0],i,ldb)].y = B[IDXC0(elems[idx].nodes[0],i,ldb)].y+temp.y;
                //atomicAdd(&B[IDXC0(elems[idx].nodes[0],i,ldb)].x,-cuCrealf(temp));
                //atomicAdd(&B[IDXC0(elems[idx].nodes[0],i,ldb)].y,-cuCimagf(temp));
                //atomicFloatComplexSub(&B[IDXC0(elems[idx].nodes[1],i,ldb)],
                //        cuCmulf(bc,gCoeffs_sgl2[j]));
                temp = cuCmulf(bc,gCoeffs_sgl2[j]);
                B[IDXC0(elems[idx].nodes[1],i,ldb)].x = B[IDXC0(elems[idx].nodes[1],i,ldb)].x+temp.x;
                B[IDXC0(elems[idx].nodes[1],i,ldb)].y = B[IDXC0(elems[idx].nodes[1],i,ldb)].y+temp.y;
                //atomicAdd(&B[IDXC0(elems[idx].nodes[1],i,ldb)].x,-cuCrealf(temp));
                //atomicAdd(&B[IDXC0(elems[idx].nodes[1],i,ldb)].y,-cuCimagf(temp));
                //atomicFloatComplexSub(&B[IDXC0(elems[idx].nodes[2],i,ldb)],
                //        cuCmulf(bc,gCoeffs_sgl3[j]));
                temp = cuCmulf(bc,gCoeffs_sgl3[j]);
                B[IDXC0(elems[idx].nodes[2],i,ldb)].x = B[IDXC0(elems[idx].nodes[2],i,ldb)].x+temp.x;
                B[IDXC0(elems[idx].nodes[2],i,ldb)].y = B[IDXC0(elems[idx].nodes[2],i,ldb)].y+temp.y;
                //atomicAdd(&B[IDXC0(elems[idx].nodes[2],i,ldb)].x,-cuCrealf(temp));
                //atomicAdd(&B[IDXC0(elems[idx].nodes[2],i,ldb)].y,-cuCimagf(temp));
            }
        }
    }
}

int atomicGenSystem_test(const float k, const triElem *elems, const int numElems, 
        const cartCoord *pnts, const int numNods, const int numCHIEF, 
        const cartCoord *srcs, const int numSrcs, cuFloatComplex *A, const int lda, 
        cuFloatComplex *B, const int ldb) {
    //Initialization of A
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int i, j;
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
    
    
    
    int xNumBlocks, xWidth = 16, yNumBlocks, yWidth = 16;
    xNumBlocks = (numNods+numCHIEF+xWidth-1)/xWidth;
    yNumBlocks = (numElems+yWidth-1)/yWidth;
    //xNumBlocks = 2;
    //yNumBlocks = 2;
    dim3 gridLayout, blockLayout;
    gridLayout.x = xNumBlocks;
    gridLayout.y = yNumBlocks;
    
    blockLayout.x = xWidth;
    blockLayout.y = yWidth;
    
    int numIter = 100;
    CUDA_CALL(cudaEventRecord(start));
    for(i=0;i<numIter;i++) {
        atomicPntsElems_nsgl_test<<<gridLayout,blockLayout>>>(k,pnts_d,numNods,0,numNods+numCHIEF-1,
            elems_d,numElems,A_d,lda,B_d,numSrcs,ldb);
        atomicPntsElems_sgl_test<<<yNumBlocks,yWidth>>>(k,pnts_d,elems_d,numElems,A_d,lda,
                B_d,numSrcs,ldb);
        
        //atomicPntsElems_nsgl<<<gridLayout,blockLayout>>>(k,pnts_d,numNods,0,numNods+numCHIEF-1,
        //    elems_d,numElems,A_d,lda,B_d,numSrcs,ldb);
        //atomicPntsElems_sgl<<<yNumBlocks,yWidth>>>(k,pnts_d,elems_d,numElems,A_d,lda,
        //        B_d,numSrcs,ldb);
    }
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaMemcpy(A,A_d,(numNods+numCHIEF)*numNods*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(B,B_d,(numNods+numCHIEF)*numSrcs*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaEventSynchronize(stop));
    float milliseconds;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds,start,stop));
    std::cout << "Elapsed system generation time: " << milliseconds/numIter << " milliseconds" << std::endl;
    CUDA_CALL(cudaFree(A_d));
    CUDA_CALL(cudaFree(B_d));
    CUDA_CALL(cudaFree(elems_d));
    CUDA_CALL(cudaFree(pnts_d));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    return EXIT_SUCCESS;
}


