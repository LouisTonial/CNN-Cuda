// INCLUDE

#include <stdlib.h>
#include <stdio.h>


// PART 1 : MATRIX MANIPULATION

// Initialization of matrix

void MatrixInit(float *M, int n, int p) { 
    for(int i = 0; i < n*p; i++){
        M[i] = 1.0 - 2*float(rand()) / float(RAND_MAX);
    }
}



// Print of matrix

void MatrixPrint(float *M, int n, int p) {
    int i = 0;
    for(int line = 0; line < n; line++){
        for(int row = 0; row < p; row++){
            printf("%f   ", M[line*p+row]);
            i++;
            if(i%p == 0)
                printf("\n");
        }
    }
    printf("\n");
}



// Addition of matrix using CPU

void MatrixAdd(float *M1, float *M2, float *Mount, int n, int p) { 
    for(int line = 0; line < n; line++){
        for(int row = 0; row < p; row++){
            Mount[line*p+row]= M1[line*p+row] + M2[line*p+row];
        }
    }
}



// Addition of matrix using GPU

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mount, int n, int p) { 
    int line = blockIdx.y * blockDim.y + threadIdx.y; 
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (line < n && row < p){
        Mount[line*p + row] = M1[line*p + row] + M2[line*p + row];
    }
}



// Multiplication of matrix using CPU

void MatrixMult(float *M1, float *M2, float *Mount, int n) {
    for(int line = 0; line < n; line++){
        for(int row = 0; row < n; row++){
           for(int k = 0; k < n; k++)
                Mount[line*n+row] += M1[line*n+k] * M2[k*n+row];
        }
    }
}



// Multiplication of matrix using GPU

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mount, int n) {
    int line = blockIdx.y * blockDim.y + threadIdx.y; 
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp_sum = 0.0;    // temporary sum 
    if (line < n && row < n){
        for (int k = 0; k < n;k ++){
            tmp_sum += M1[line*n + k]*M2[k*n + row];
            Mount[line*n + row] = tmp_sum;
        }
    }
}





// MAIN

int main(int argc, char*argv[]) {
    
    // PARAMETERS INITIALISATION
    
    float *mat1, *mat2, *mat3, *mat4;
    float *cuda_mat1, *cuda_mat2, *cuda_mat3, *cuda_mat4;
    
    if(argc != 4){
        return EXIT_FAILURE;
    }
    
    int n = atoi(argv[2]);
    int p = atoi(argv[3]);



    // INITIALIZATION

    // Memory allocation for the CPU
    mat1 = (float*)malloc(n*p*sizeof(float));
    mat2 = (float*)malloc(n*p*sizeof(float));
    mat3 = (float*)malloc(n*n*sizeof(float));   // Addition matrix 
    mat4 = (float*)malloc(n*p*sizeof(float));   // Multiplication matrix
    
    // Matrix initialization
    MatrixInit(mat1, n, p);  
    MatrixInit(mat2, n, p);

    // Memory allocation for the GPU
    cudaMalloc((void**)&cuda_mat1, sizeof(float)*n*p);
    cudaMalloc((void**)&cuda_mat2, sizeof(float)*n*p);
    cudaMalloc((void**)&cuda_mat3, sizeof(float)*n*p);
    cudaMalloc((void**)&cuda_mat4, sizeof(float)*n*n);

    // Copy of the data from the CPU to the GPU
    cudaMemcpy(cuda_mat1, mat1, sizeof(float)*n*p, cudaMemcpyHostToDevice); 
    cudaMemcpy(cuda_mat2, mat2, sizeof(float)*n*p, cudaMemcpyHostToDevice);



     // CPU CALCULATION

    if(strcmp(argv[1], "cpu") == 0){
        
        // Addition
        printf("Addition result with CPU :\n\n"); 
        MatrixAdd(mat1, mat2, mat3, n, p);
        MatrixPrint(mat3, n, p);
        
        // Multiplication
        printf("Multiplication result with CPU :\n\n"); 
        MatrixMult(mat1, mat2, mat4, n);
        MatrixPrint(mat4, n, p);
    }



    // GPU CALCULATION

    dim3 block_dim(n,p);
    dim3 grid_dim(1,1);

    if(strcmp(argv[1], "gpu") == 0){
        
        // Addition
        printf("Addition result with GPU :\n\n"); 
        cudaMatrixAdd<<<block_dim, grid_dim>>>(cuda_mat1, cuda_mat2, cuda_mat3, n, p);
        cudaDeviceSynchronize();
        cudaMemcpy(mat3, cuda_mat3, sizeof(float)*n*p, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        MatrixPrint(mat3, n, p);
        
        // Multiplication
        printf("Multiplication result with GPU :\n\n");
        cudaMatrixMult<<<block_dim, grid_dim>>>(cuda_mat1, cuda_mat2, cuda_mat4, n);
        cudaDeviceSynchronize();
        cudaMemcpy(mat4, cuda_mat4, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        MatrixPrint(mat4, n, n);
    }


   
    // MEMORY FREEING
    free(mat1);
    free(mat2);
    free(mat3);
    free(mat4);
    cudaFree(cuda_mat1);
    cudaFree(cuda_mat2);
    cudaFree(cuda_mat3);
    cudaFree(cuda_mat4);

    return 0;
}
