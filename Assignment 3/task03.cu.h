
/*
Task 3.c
*/
template<class T>
__global__ void naive_matrixmult(float* matrix1, float* matrix2, float* ret_matrix, int rows, int cols, int cols2){
    T res = 0.0;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(j>rows && i>cols2){
        return;
    }
    
    for(int k = 0; k<cols;k++){
        res = res + matrix1[j*cols+k] * matrix2[k*cols2+i];
    }
    ret_matrix[j*cols+i] = res;
}
