__global__ void quick_2c_kernel(float* matrix_in, float* matrix_out, int row){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int col = 63;
    if(i >= rows){
        return ;
    }
    float accum = matrix_in[i*row] * matrix_in[i*row];
    matrix_out[i*row+0] = accum;
    for(int k=1; k<64; k++){
        float* tmpA = matrix_in[i*row+k]
        accum = sqrt(accum)+tmpA * tmpA;
        matrix_out[i*row+k]
    }
}
