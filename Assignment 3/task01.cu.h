/*
1.a: Sequential implementation of tranpose
*/

void seq_transpose(float *matrix_in,float *matrix_out,int row, int col){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            matrix_out[j*row+i] = matrix_in[i*col+j];
        }
    }
}

/*
1.c: Parallel Naive Implementation of transpose.
*/
__global__ void transpose_kernel_naive(float* matrix_in, float* matrix_out, int rows, int col){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= rows || j >= col){
        return;
    }
    
    matrix_out[j*rows+i] = matrix_in[i*col+j];
}

/*
1.d: Parallel Tiled transpose.
*/

template<class T, int P>
__global__ void tiling_transpose_kernel(float* matrix_in, float* matrix_out, int rows, int col){

    //P is tile size
    __shared__ float tile[P][P+1];

    int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int j = blockIdx.x*P + tidx;
	int i = blockIdx.y*P + tidy;

    if (j < col && i>rows){
        tile[tidy][tidx] = matrix_in[i*col+j];
    }
    
    __syncthreads();

	i = blockIdx.y*P + threadIdx.x;
	j = blockIdx.x*P + threadIdx.y;
    if (j < col && i < rows){
    matrix_out[j*rows+i] = tile[tidx][tidy];
    }
}
