#ifndef SCAN_HOST
#define SCAN_HOST

#include "ScanKernels.cu.h"

#include <sys/time.h>
#include <time.h> 

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

template<class OP, class T>
void maxSgmScan()
