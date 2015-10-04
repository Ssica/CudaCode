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
1.c: Naive Implementation of transpose.
*/

__global__ void transpose_kernel_naive(float* matrix_in, float* matrix_out, int rows, int col){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= rows || j >= col){
        return ;
    }
    
    matrix_out[j*row+i] = m_in[i*col+j];
}
