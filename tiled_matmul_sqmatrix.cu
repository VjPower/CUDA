#include <iostream>
#include <cuda_runtime.h>

#define N 4
#define TILE 2

using namespace std; 

__global__ void matmul(float *A, float *B, float *C){

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    if(row > N || col > N){
        return;
    }

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum;

    for(int i =0; i< N/TILE; i++){
        As[ty][tx] = A[row*N + (i*TILE + tx)];
        Bs[ty][tx] = B[(i*TILE + ty)*N + col];

        __syncthreads();

        sum = 0;
        for(int j=0; j<TILE; j++){
            sum += As[ty][j] * Bs[j][tx];
        }

        C[row*N + col] +=sum;

        __syncthreads();

    }
}

int main(){

    size_t size = N*N*sizeof(float);

    float *hA = new float[N*N];
    float *hB = new float[N*N];
    float *hC = new float[N*N]();

    for (int i=0; i<N*N; i++){
        hA[i] = i%3;
        hB[i] = i%4;
    }

    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            cout<<hA[i*N + j]<<" ";
        }
        cout<<endl;
    }
    cout<<"++++"<<endl;
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            cout<<hB[i*N + j]<<" ";
        }
        cout<<endl;
    }

    float *dA, *dB, *dC; 
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
    
    cudaMemset(dC, 0, size);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid(N/TILE, N/TILE);

    matmul<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    cout<<"==============";
    for (int i=0; i<N*N; i++){
        if(i%4==0){
            cout<<endl;
        }
        cout<<hC[i]<< " ";
    }

    delete[] hA; 
    delete[] hB;
    delete[] hC;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

}