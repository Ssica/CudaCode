#include "task01.cu.h"
#include "task02.cu.h"
#include "task03.cu.h"
#include "seq.h"

#define ROWS 8192
#define COLS 8192
#define TILE 32

int main(){
size_t size = COLS * ROWS;
	size_t mem_size = sizeof(float) * size;
	float* h_A = (float*) malloc(mem_size);
	float* h_B = (float*) malloc(mem_size);
	float* h_C = (float*) malloc(mem_size);

    float* d_A;
	float* d_C;
	cudaMalloc((void**)&d_A,mem_size);
	cudaMalloc((void**)&d_C,mem_size);

    srand(time(0));
   init_matrix(h_A,size);
}
