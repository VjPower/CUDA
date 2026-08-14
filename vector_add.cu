#include <iostream>
#include <cuda_runtime.h>

using namespace std; 

__global__ void vector_add(int* dA, int* dB, int* dC, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        dC[i] = dA[i] + dB[i];
    }
}
int main(){

    //Initialisations on the Host system
    int n = 10000;
    size_t size = n*sizeof(int);

    int* hA= new int[n];
    int* hB= new int[n];
    int* hC= new int[n];

    for (int i=0; i<n; i++){
        hA[i] = i%3;
        hB[i] = i%4;
    }

    // Allocating memory on the GPU
    int *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    //Copy data from Host to GPU 
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, size, cudaMemcpyHostToDevice);

    //Launching the kernel 
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1)/threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, n);

    //Copy data back to host from GPU
    cudaMemcpy(hA, dA, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hB, dB, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
    
    for(int i=0; i<10; i++){
        cout<<hC[i]<<endl;
    }

    //Free GPU memory 
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    //Free Host memory 
    free(hA);
    free(hB);
    free(hC);

}