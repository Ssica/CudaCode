/*
2.c
*/

__global__ void quick_2c_kernel(float* matrix_in, float* matrix_out, int row){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    //int j = blockIdx.x * blockDim.x + threadIdx.x;
    int col = 64;
    if(i >= row){
        return ;
    }

    i = i*col;
    float accum = matrix_in[i] * matrix_in[i];
    matrix_out[i+0] = accum;
    for(int k=1; k<64; k++){
        float tmpA = matrix_in[i+k];
        accum = sqrt(accum)+tmpA * tmpA;
        matrix_out[i+k];
    }
}


/*
2.d
*/

__global__ void 2d_kernel(float* matrix_in, float* matrix_out, int row){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    //int j = blockIdx.x * blockDim.x + threadIdx.x;
    int col = 64;
    if(i >= row){
        return ;
    }
    i = i+row;
    float accum = matrix_in[i] * matrix_in[i];
    matrix_out[i+0] = accum;
    for(int k=1; k<row; k++){
        i = i + row;
        float* tmpA = matrix_in[i];
        accum = sqrt(accum)+tmpA * tmpA;
        matrix_out[i+k];
    }
}



