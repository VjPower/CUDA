#include <iostream>
#include <cuda_runtime.h>

using namespace std; 

#define TILE 2

#define X 6 
#define Y 4
#define Z 4 

__global__ void matmul(float *A, float* B, float* C){
    int ty= threadIdx.y;
    int tx= threadIdx.x;

    int row = blockIdx.y*TILE + ty;
    int col = blockIdx.x*TILE + tx;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int TILE_COUNT = (Y+TILE-1)/TILE;

    for(int i=0; i<TILE_COUNT; i++){
        int ACol = i*TILE + tx;
        int BRow = i*TILE + ty;

        As[ty][tx] = (row < X || ACol < Y) ? A[row*Y + ACol] : 0.0f;
        Bs[ty][tx] = (BRow < Y || col < Z) ? B[BRow*Z + col] : 0.0f; 

        __syncthreads();

        float sum = 0;
        for (int j=0; j<TILE; j++){
            sum += As[ty][j] * Bs[j][tx];
        }
        __syncthreads();

        if(row< X || col < Z){
            C[row*Z + col] +=sum;
        }
    }
}

int main(){
    float *hA = new float[X*Y];
    float *hB = new float[Y*Z];
    float *hC = new float[X*Z]();

    float *dA, *dB, *dC;

    for (int i=0; i<X*Y; i++){
        hA[i] = i%3;
    }
    for (int i=0; i<Y*Z; i++){
        hB[i] = i%3;
    }

    for (int i=0;i<X;i++){
        for (int j=0;j<Y;j++){
            cout<<hA[i*Y + j]<<" ";
        }
        cout<<endl;
    }
    cout<<"++++"<<endl;
    for (int i=0;i<Y;i++){
        for (int j=0;j<Z;j++){
            cout<<hB[i*Z + j]<<" ";
        }
        cout<<endl;
    }

    cudaMalloc(&dA, X*Y*sizeof(float));
    cudaMalloc(&dB, Y*Z*sizeof(float));
    cudaMalloc(&dC, X*Z*sizeof(float));

    cudaMemset(dC, 0, X*Z*sizeof(float));

    cudaMemcpy(dA, hA, X*Y*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, Y*Z*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((Z+threadsPerBlock.x -1)/threadsPerBlock.x, (X+threadsPerBlock.y-1)/threadsPerBlock.y);

    matmul<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC);

    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, X*Z*sizeof(float), cudaMemcpyDeviceToHost);

    cout<<"==============";

    for (int i=0; i<X*Z; i++){
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