#include <iostream>
#include <cuda_runtime.h>

using namespace std; 

#define M 4
#define N 3
#define L 2

__global__ void naiveMul(float *A, float *B, float *C){
    int row = blockIdx.y* blockDim.y + threadIdx.y;
    int col = blockIdx.x* blockDim.x + threadIdx.x;

    if(row < M && col < L){
        for(int i=0; i<N; i++){
            C[row*L + col] += A[row*N + i] * B[i*L + col];
        }
    }
}
int main(){
    float *hA = new float[M*N];
    float *hB = new float[N*L];
    float *hC = new float[M*L];

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            hA[i*N + j] = (i*j)%3;
            cout<<hA[i*N + j]<<" ";
        }
        cout<<endl;
    }
    cout<<"=============="<<endl;

    for (int i=0; i<N;i++){
        for(int j=0; j<L; j++){
            hB[i*L + j] = (i*j)%3;
            cout<<hB[i*L + j]<<" ";
        }
        cout<<endl;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, M*N*sizeof(float));
    cudaMalloc(&dB, N*L*sizeof(float));
    cudaMalloc(&dC, M*L*sizeof(float));

    cudaMemcpy(dA, hA, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N*L*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M*L*sizeof(float)); // cudaMemset allocates 0 (the 2nd argument value) to every respective allocated Byte in the memory 

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((L + threadsPerBlock.x -1)/threadsPerBlock.x, (M + threadsPerBlock.y - 1)/threadsPerBlock.y);

    naiveMul<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC);

    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, M*L*sizeof(float), cudaMemcpyDeviceToHost);

    cout<<"====OUTPUT===="<<endl;

    for (int i = 0; i<M; i++){
        for (int j = 0; j<L; j++){
            cout<<hC[i*L+j]<<" ";
        }
        cout<<endl;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] hA; 
    delete[] hB;
    delete[] hC;
}