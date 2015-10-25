#include "task01.cu.h"
#include "task02.cu.h"
#include "task03.cu.h"
#include "seq.h"

#define ROWS 8192
#define COLS 8192
#define TILE 32
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1){
	unsigned int resolution=1000000;
	long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
	result->tv_sec = diff / resolution;
	result->tv_usec = diff % resolution;
	return (diff<0);
}


void validate_trans(float* mat1, float* mat2){
    int bo = 0;    
    for(int i=0; i<ROWS; i++){
        for(int j=0; j<COLS; j++){
            if (mat1[i*COLS+j] != mat2[j*ROWS+i] ){
                bo = 1;
            } 
        }   
    }
    if(bo == 1){
        printf("validate of transpose failed");
    }
    printf("validate of transpose correct");
}
int main(){
    size_t size = COLS * ROWS;
	size_t mem_size = sizeof(float) * size;
	float* h_A = (float*) malloc(mem_size);
	float* h_B = (float*) malloc(mem_size);
	float* h_C = (float*) malloc(mem_size);

    int const T = 32;
    int dimx = (COLS+T-1)/T;
    int dimy = (ROWS+T-1)/T;
    dim3 block(T,T,1), grid(dimx, dimy, 1);

    unsigned long int elapsed;
    srand(time(0));

    for(int i = 0; i<ROWS; i++){
        for(int j = 0; j<COLS; j++){
            h_A[i*ROWS+j] = i;
        }
    }
    struct timeval t_start,t_end,t_diff;
    //Sequential transpose
	gettimeofday(&t_start, NULL);
    seque_transpose(h_A, h_B, ROWS, COLS);
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);	
	printf("Sequential transpose ran in %lu microseconds.\n",elapsed);
    validate_trans(h_A, h_B);

    //Parallel tranpose
    float* d_A;
	float* d_C;
	cudaMalloc((void**)&d_A,mem_size);
	cudaMalloc((void**)&d_C,mem_size);

    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
    gettimeofday(&t_start,NULL);
    transpose_kernel_naive<<<grid,block>>>(d_A, d_C, ROWS, COLS);
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Naive Parallel Transpose Kernel ran in %lu microseconds.\n",elapsed);   
    validate_trans(d_A, d_C);
    cudaFree(d_A);    
    cudaFree(d_C);


	cudaMalloc((void**)&d_A,mem_size);
	cudaMalloc((void**)&d_C,mem_size);
    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
    gettimeofday(&t_start,NULL);
    tiling_transpose_kernel<T, TILE><<<grid,block>>>(d_A, d_C, ROWS, COLS);
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Tiled Parallel Transpose Kernel ran in %lu microseconds.\n",elapsed);
    cudaFree(d_A);    
    cudaFree(d_C);

    //Task 2

	cudaMalloc((void**)&d_A,mem_size);
	cudaMalloc((void**)&d_C,mem_size);
    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
    gettimeofday(&t_start,NULL);
    quick_2c_kernel<<<grid, block>>>(d_A, d_C, ROWS, COLS);
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("2.c Kernel ran in %lu microseconds.\n",elapsed);
    cudaFree(d_A);    
    cudaFree(d_C);

	cudaMalloc((void**)&d_A,mem_size);
	cudaMalloc((void**)&d_C,mem_size);
    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
    gettimeofday(&t_start,NULL);
    quick_2c_kernel(d_A, d_C, ROWS);
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("2.d Kernel ran in %lu microseconds.\n",elapsed);
    cudaFree(d_A);    
    cudaFree(d_C);

    //Matrix Multiplication

	gettimeofday(&t_start, NULL);    
    matrixmult_seq(h_A, h_A, h_B, ROWS, COLS, COLS);
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Sequential Matrix Multiplaction ran in %lu microseconds.\n",elapsed);

    //Parallel Matrix Multiplication
	cudaMalloc((void**)&d_A,mem_size);
	cudaMalloc((void**)&d_C,mem_size);
    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
    gettimeofday(&t_start,NULL);
    naive_matrixmult<T><<<grid,block>>>(d_A, d_A, d_C, ROWS, COLS, COLS);
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Naive Matrix Multiplication Kernel ran in %lu microseconds.\n",elapsed);
    cudaFree(d_A);    
    cudaFree(d_C);


}
