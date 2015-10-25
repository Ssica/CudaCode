#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<time.h>

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1){
	unsigned int resolution=1000000;
	long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
	result->tv_sec = diff / resolution;
	result->tv_usec = diff % resolution;
	return (diff<0);
}

/*
1.a: Sequential implementation of tranpose
*/

void seque_transpose(float *matrix_in,float *matrix_out,int row, int col){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            matrix_out[j*row+i] = matrix_in[i*col+j];
        }
    }
}


/*
Task 3.a
*/

void matrixmult_seq(float* matrix1, float* matrix2, float* ret_matrix, int rows, int cols,int cols2){

 for(int i=0; i<rows; i++){

    for(int j = 0; j<cols2;j++){
        float res = 0.0;
        
        for(int k = 0; k<cols; k++){
            res = res + matrix1[i*cols+k] * matrix2[k*cols2+j];
        }
        ret_matrix[i*cols2+j] = res;    
    }  
 }   
}
