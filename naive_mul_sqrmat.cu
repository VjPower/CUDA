#include <iostream>
#include <cuda_runtime.h>
using namespace std; 


__global__ void naiveMul(int *A, int *B, int *C, int N){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < N && col < N){
        for(int i=0; i<N; i++){
            C[row*N + col] += A[row*N + i] * B[i*N + col];
        }
    }
}


int main(){

    int n = 4;
    size_t size = n*n * sizeof(int);

    int *hA= new int[n*n];
    int *hB= new int[n*n];
    int *hC= new int[n*n](); //The () at the end tells the computer to set each value in the allocated space to 0

    for (int i = 0; i<n; i++){
        for (int j = 0; j<n; j++){
            hA[i*n + j] = i%3 + j;
            hB[i*n + j] = j%3 + i;
        }
    }
    // for (int i = 0; i<n; i++){
    //     for (int j = 0; j<n; j++){
    //         cout<<hA[i*n+j]<<" ";
    //     }
    //     cout<<endl;
    // }
    // cout<<"========"<<endl;
    // for (int i = 0; i<n; i++){
    //     for (int j = 0; j<n; j++){
    //         cout<<hB[i*n+j]<<" ";
    //     }
    //     cout<<endl;
    // }
    // cout<<"========"<<endl;


    int *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1)/threadsPerBlock.x, (n + threadsPerBlock.y - 1)/threadsPerBlock.y);

    naiveMul<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();
    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);


    for (int i = 0; i<n; i++){
        for (int j = 0; j<n; j++){
            cout<<hC[i*n+j]<<" ";
        }
        cout<<endl;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] hA;    //free() can only be used with malloc
    delete[] hB;    //delete[] is used with "new __"
    delete[] hC;



}