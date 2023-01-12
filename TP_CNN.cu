// INCLUDE

#include <stdlib.h>
#include <stdio.h>



// PART 2 : Première couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

// Layer 1 - Génération des données de test

// Initialization of matrix function

void MatrixInit(float *M, int type, int k, int l, int m ) { 
    if (type == 0){
        for (int i = 0; i < k*l*m; i++){
            M[i] = 0;
        }
    }

    if (type == 1){
        for (int i = 0; i < k*l*m; i++){
            M[i] = 1;
        }
    }

    else if (type == 2){
        for (int i = 0; i < k*l*m; i++){
            M[i] = 1.0 -(2*float(rand())) / float(RAND_MAX);;
        }
    }
}



// Print of matrix function

void MatrixPrint(float *M, int n, int p) {
    for(int line = 0; line < n; line++){
        for(int row = line*p; row < p*(line+1); row++){
            printf("%f   ", M[row]);
        }
        printf("\n");
    }
    printf("\n");
}




// Layer 2 - Convolution

// 2D Convolution function

__global__ void cudaConv2D(float* M, float* kernel, float* Mount, int M_line, int M_row, int kernel_size, int Mount_line, int Mount_row, int nb_kernel){
   
    int line = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp_sum;

    if (line < Mount_line && row < Mount_row){ 

        int tot_kernel = kernel_size * kernel_size;
        int tot_Mount = Mount_line * Mount_row;

        for (int n_k = 0; n_k < nb_kernel; n_k++){
            tmp_sum = 0.0;
            for (int kernel_line = 0; kernel_line < kernel_size; kernel_line++) {
                for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                    tmp_sum += M[(line + kernel_line) * M_row + (row + kernel_row)] * kernel[kernel_row * kernel_size + kernel_row + n_k * tot_kernel];  
                }
            }
            Mount[line * Mount_row + row + n_k * tot_Mount] = tmp_sum;
        }
    }
}




// Layer 3 - 2D Subsampling

// 2D Subsampling function

__global__ void cudaSubsampling(float* M, float* Mount, int M_line, int M_row, int M_prof, int meanpool_size, int Mount_line, int Mount_row){

    int line = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (line % meanpool_size == 0 && row % meanpool_size == 0){ 

        float tmp_sum;
        int tot_meanpool = meanpool_size * meanpool_size;
        int tot_M = M_line * M_row;
        int tot_Mount = Mount_line * Mount_row;

        for (int n_prof = 0; n_prof < M_prof; n_prof++){
            tmp_sum = 0.0;
            for (int meanpool_line = 0; meanpool_line < meanpool_size; meanpool_line++) {
                for (int meanpool_row = 0; meanpool_row < meanpool_size; meanpool_row++) {
                    tmp_sum += M[(line + meanpool_line) * M_row + row + meanpool_row + n_prof * tot_M] / tot_meanpool;
                }
            }

            if (line == 0){
                Mount[line * Mount_row + (row / meanpool_size) + n_prof * tot_Mount] = tmp_sum;
            }

            else if (row == 0){
                Mount[(line / meanpool_size) * Mount_row + row + n_prof * tot_Mount] = tmp_sum;
            }

            else{
                Mount[(line / meanpool_size) * Mount_row + (row / meanpool_size) + n_prof * tot_Mount] = tmp_sum;
            }
        }
    }
}




// Layer 4 - Activation function

// Activation 

__device__ float* activation_tanh(float* M, int M_line, int M_row, int M_prof){

    int line = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (line < M_line && row < M_row){
        int tot_M = M_line * M_row;
        for (int n_prof = 0; n_prof < M_prof; n_prof++){
            M[line * M_row + row + n_prof * tot_M] = tanh(M[line * M_row + row + n_prof * tot_M]);
        }
    }
    return M;
}


__global__ void cudaTanh(float* M, int M_line, int M_row, int M_prof){
    activation_tanh(M, M_line, M_row, M_prof);
}






// MAIN

int main(int argc, char*argv[]) {
    

    // Initialization
    float *raw_data, *C1_data, *S1_data, *C1_kernel, *C2_data, *S2_data, *W1_kernel, *B1_kernel, *D1_data;
    float *cuda_raw_data, *cuda_C1_data, *cuda_S1_data, *cuda_C1_kernel, *cuda_C2_data, *cuda_S2_data, *cuda_W1_kernel, *cuda_B1_kernel, *cuda_D1_data ;


    // Memory allocation for the CPU
    raw_data = (float*)malloc(1*32*32*sizeof(float));
    C1_kernel = (float*)malloc(6*5*5*sizeof(float));
    C1_data = (float*)malloc(6*28*28*sizeof(float));
    S1_data = (float*)malloc(6*14*14*sizeof(float));
    C2_data = (float*)malloc(6*10*10*sizeof(float));
    S2_data = (float*)malloc(6*5*5*sizeof(float));
    W1_kernel = (float*)malloc(120*400*sizeof(float));
    B1_kernel = (float*)malloc(120*sizeof(float));
    D1_data = (float*)malloc(120*sizeof(float));
    

    // Matrix initialization
    MatrixInit(raw_data, 2, 1, 32, 32);
    MatrixInit(C1_kernel, 2, 6, 5, 5);  
    MatrixInit(C1_data, 0, 6, 28, 28);
    MatrixInit(S1_data, 0, 6, 14, 14);  
    MatrixInit(C2_data, 0, 6, 10, 10);  
    MatrixInit(S2_data, 0, 6, 5, 5);
    MatrixInit(W1_kernel, 1, 1, 400, 120);  
    MatrixInit(B1_kernel, 1, 1, 1, 120);
    MatrixInit(D1_data, 1, 1, 1, 120);


    // Memory allocation for the GPU
    cudaMalloc((void**)&cuda_raw_data, sizeof(float)*1*32*32);
    cudaMalloc((void**)&cuda_C1_kernel, sizeof(float)*6*5*5);
    cudaMalloc((void**)&cuda_C1_data, sizeof(float)*6*28*28);
    cudaMalloc((void**)&cuda_S1_data, sizeof(float)*6*14*14);
    cudaMalloc((void**)&cuda_C2_data, sizeof(float)*6*10*10);
    cudaMalloc((void**)&cuda_S2_data, sizeof(float)*6*5*5);
    cudaMalloc((void**)&cuda_W1_kernel, sizeof(float)*120*400);
    cudaMalloc((void**)&cuda_B1_kernel, sizeof(float)*120);
    cudaMalloc((void**)&cuda_D1_data, sizeof(float)*120);


    // Copy of the data from the CPU to the GPU
    cudaMemcpy(cuda_raw_data, raw_data, sizeof(float)*1*32*32, cudaMemcpyHostToDevice); 
    cudaMemcpy(cuda_C1_kernel, C1_kernel, sizeof(float)*6*5*5, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_C1_data, C1_data, sizeof(float)*6*28*28, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_S1_data, S1_data, sizeof(float)*6*14*14, cudaMemcpyHostToDevice); 
    cudaMemcpy(cuda_C2_data, C2_data, sizeof(float)*6*10*10, cudaMemcpyHostToDevice); 
    cudaMemcpy(cuda_S2_data, S2_data, sizeof(float)*6*5*5, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_W1_kernel, W1_kernel, sizeof(float)*120*400, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B1_kernel, B1_kernel, sizeof(float)*120, cudaMemcpyHostToDevice); 
    cudaMemcpy(cuda_D1_data, D1_data, sizeof(float)*120, cudaMemcpyHostToDevice);


    // Layers
    dim3 block_dim(32,32);
    dim3 grid_dim(1,1);


    cudaConv2D<<<block_dim, grid_dim>>>(cuda_raw_data, cuda_C1_kernel, cuda_C1_data, 32, 32, 5, 6, 28, 28);
    cudaDeviceSynchronize();

    cudaTanh<<<block_dim, grid_dim>>>(cuda_C1_data, 28, 28, 6);
    cudaDeviceSynchronize();

    cudaSubsampling<<<block_dim, grid_dim>>>(cuda_C1_data, cuda_S1_data, 28, 28, 6, 2, 14, 14);
    cudaDeviceSynchronize();

    cudaConv2D<<<block_dim, grid_dim>>>(cuda_S1_data, cuda_C1_kernel, cuda_C2_data, 14, 14, 15, 16, 10, 10);
    cudaDeviceSynchronize();

    cudaTanh<<<block_dim, grid_dim>>>(cuda_C2_data, 10, 10, 16);
    cudaDeviceSynchronize();

    cudaSubsampling<<<block_dim, grid_dim>>>(cuda_C2_data, cuda_S2_data, 10, 10, 16, 2, 5, 5);
    cudaDeviceSynchronize();


    // Copy of the data from the GPU to the CPU
    cudaMemcpy(cuda_C1_data, C1_data, sizeof(float)*6*28*28, cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_S1_data, S1_data, sizeof(float)*6*14*14, cudaMemcpyDeviceToHost); 
    cudaMemcpy(cuda_C2_data, C2_data, sizeof(float)*6*10*10, cudaMemcpyDeviceToHost); 
    cudaMemcpy(cuda_S2_data, S2_data, sizeof(float)*6*5*5, cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_D1_data, D1_data, sizeof(float)*120, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    // Print
    printf("Initial matrix :\n\n");
    MatrixPrint(raw_data, 32, 32);

    printf("Convolution kernel :\n\n");
    MatrixPrint(C1_kernel, 5, 5);

    printf("First 2D Convolution & tanh result :\n\n");
    cudaMemcpy(C1_data, cuda_C1_data, sizeof(float)*6*28*28, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    MatrixPrint(C1_data, 28, 28);

    printf("First Meanpooling result :\n\n");
    cudaMemcpy(S1_data, cuda_S1_data, sizeof(float)*6*14*14, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    MatrixPrint(S1_data, 14, 14);

    printf("Second 2D Convolution & tanh result :\n\n");
    cudaMemcpy(C2_data, cuda_C2_data, sizeof(float)*6*10*10, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    MatrixPrint(C2_data, 10, 10);

    printf("Second Meanpooling result :\n\n");
    cudaMemcpy(S2_data, cuda_S2_data, sizeof(float)*6*5*5, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    MatrixPrint(S2_data, 5, 5);


    // Memory freeing
    free(raw_data);
    free(C1_kernel);
    free(C1_data);
    free(S1_data);
    free(C2_data);
    free(S2_data);
    free(W1_kernel);
    free(B1_kernel); 
    free(D1_data);
    
    cudaFree(cuda_raw_data);
    cudaFree(cuda_C1_kernel);
    cudaFree(cuda_C1_data);
    cudaFree(cuda_S1_data);
    cudaFree(cuda_C2_data);
    cudaFree(cuda_S2_data);
    cudaFree(cuda_W1_kernel);
    cudaFree(cuda_B1_kernel);
    cudaFree(cuda_D1_data);

    return 0;

}
