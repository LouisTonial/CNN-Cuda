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



// Addition of matrix

void MatrixAdd(float *M1, float *M2, float *Mount, int n, int p) {   
    for(int line = 0; line < n; line++){
        for(int row = 0; row < p; row++){
            Mount[line*p+row]= M1[line*p+row] + M2[line*p+row];
        }
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mount, int n, int p) {
    
}



// Multiplication of matrix using CPU

void MatrixMult(float *M1, float *M2, float *Mount, int n) {
    for(int line = 0; line < n; line++){
        for(int row = 0; row < n; row++){
            Mount[0] = 0;
            for(int k = 0; k < n; k++)
                Mount[line*n+row] += M1[line*n+k] * M2[k*n+row];
        }
    }
}



// Multiplication of matrix using GPU

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mount, int n) {
    
}





// MAIN

int main(int argc, char*argv[]) {
    int n = 3; // Size of the matrix
    int p = 3; // Size of the matrix
    
    float *mat1;
    mat1 = (float*)malloc(n*p*sizeof(float));
    MatrixInit(mat1, n, p);
    if(mat1 == NULL){ // We check if the allocation worked or not
        exit(0); // We stop
    }
    MatrixPrint(mat1, n, p);
    
    float *mat2;
    mat2 = (float*)malloc(n*p*sizeof(float));
    MatrixInit(mat2, n, p);
    if(mat2 == NULL){ // We check if the allocation worked or not
        exit(0); // We stop
    }
    MatrixPrint(mat2, n, p);
    
    float *mat3;
    mat3 = (float*)malloc(n*n*sizeof(float));
    MatrixAdd(mat1, mat2, mat3, n, p);
    if(mat3 == NULL){ // We check if the allocation worked or not
        exit(0); // We stop
    }
    MatrixPrint(mat3, n, p);

    float *mat4;
    mat4 = (float*)malloc(n*p*sizeof(float));
    MatrixMult(mat1, mat2, mat4, n);
    if(mat4 == NULL){ // We check if the allocation worked or not
        exit(0); // We stop
    }
    MatrixPrint(mat4, n, p);

    free(mat1);
    free(mat2);
    free(mat3);
    free(mat4);

    return 0;
}
