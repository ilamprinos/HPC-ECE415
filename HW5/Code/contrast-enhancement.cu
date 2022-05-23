#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "hist-equ.h"

#define nbr_bin 256
#define HIST_BLOCK 16
#define RESULT_BLOCK 1024

#define CudaErrorCheck() \
    error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        printf("Cuda error %s: %d: '%s'.\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        cudaFree(lut); \
        cudaFree(hist); \
        cudaFree(source_image); \
        cudaFree(result.img); \
        cudaDeviceReset(); \
        exit(1); \
    }


__global__ void histogram_gpu( int *hist_out, unsigned char *img_in, int img_size){
	
	__shared__ int shist[256];
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = blockDim.x*gridDim.x;
	int idx = threadIdx.x + threadIdx.y*blockDim.x;
	int image_idx = x +y*offset;

	shist[idx] = 0;

	if(image_idx < img_size){
		__syncthreads();
		atomicAdd(&shist[img_in[image_idx]], 1);
	}

	__syncthreads();

	atomicAdd(&hist_out[idx], shist[idx]);

}

__global__ void lut_gpu(int *lut, int *hist, int img_size, int min){


    int idx  = threadIdx.x, d;
    __shared__ int scdf[nbr_bin];

    scdf[idx] = hist[idx];

    for( d = 1; d < nbr_bin; d = d << 1){
    	__syncthreads();

    	if( idx < (nbr_bin-d) ){
    		scdf[idx + d] += scdf[idx];
    	}
    }

    __syncthreads();

    lut[idx] = (int)(((float)scdf[idx] - min)*255/(img_size - min) + 0.5);

}


__global__ void image_result(unsigned char * img_out, int *lut, unsigned char * img_in, int img_size){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    int reg = lut[img_in[idx]];

    if (idx < img_size){
    	if(reg > 255){
        	img_out[idx] = 255;
   		}
    	else if(reg < 0){
        	img_out[idx] = 0;
    	}
    	else{
    		img_out[idx] = (unsigned char)reg;
    	}
    }

}


PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    //int hist[256];
    int d, *lut, *hist;
    unsigned char *source_image;
    cudaError_t error;
    dim3 hist_grid, hist_block, lut_grid, lut_block, res_grid, res_block;
    double min_time, total_time;
    struct timespec  tv1, tv2;
    cudaEvent_t start, stop, begin, finish;
    
    result.w = img_in.w;
    result.h = img_in.h;

    cudaMallocManaged((unsigned char**)&result.img, img_in.h * img_in.w*sizeof(unsigned char));
    cudaMalloc((void**)&source_image, img_in.h * img_in.w*sizeof(unsigned char));
    cudaMallocManaged((void**)&hist, nbr_bin*sizeof(int));
    cudaMalloc((void**)&lut, nbr_bin*sizeof(int));

     if( (source_image==NULL) || (lut==NULL) || (result.img==NULL) || (hist==NULL) ){
        printf("Error while allocating memory.\n");
        exit(1);
    }

    hist_block.x = HIST_BLOCK ;
    hist_block.y = HIST_BLOCK ;

    if( img_in.w %  hist_block.x == 0){
    	d = 0;
    }
    else{ d = 1; }

    hist_grid.x = img_in.w/hist_block.x + d;

	if( img_in.h %  hist_block.y == 0){
    	d = 0;
    }
    else{ d = 1; }

    hist_grid.y = img_in.h/hist_block.y + d; 

    cudaEventCreate(&begin);
    cudaEventCreate(&finish);

    cudaEventRecord(begin); 

    cudaMemset(hist, 0, nbr_bin*sizeof(int));
    CudaErrorCheck();

    cudaMemcpy(source_image, img_in.img, img_in.h * img_in.w * sizeof(unsigned char), cudaMemcpyHostToDevice);
    CudaErrorCheck();

    histogram_gpu<<<hist_grid, hist_block>>>(hist, source_image, img_in.h * img_in.w);
    cudaDeviceSynchronize();
    CudaErrorCheck();

    cudaEventRecord(finish);

    cudaEventSynchronize(finish);

    float elapsed1 = 0;
    cudaEventElapsedTime(&elapsed1, begin, finish);  

    printf("Histogram time = %.9f sec \n", elapsed1 / 1000);

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    int min = 0;
    int i = 0;
    while(min == 0){
        min = hist[i++];
    }
 	
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    min_time = (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 + (double) (tv2.tv_sec - tv1.tv_sec);

    lut_grid.x = 1;
    lut_block.x = 256;

    if((img_in.h * img_in.w) % RESULT_BLOCK == 0){
    	d = 0;
    }
    else{ d = 1; }

    res_grid.x = (img_in.h * img_in.w) /RESULT_BLOCK + d;
    res_block.x = RESULT_BLOCK;  

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); 

    lut_gpu<<<lut_grid, lut_block>>>(lut, hist, img_in.h * img_in.w, min);
    cudaDeviceSynchronize();
    CudaErrorCheck();

    image_result<<<res_grid, res_block >>>(result.img, lut, source_image, img_in.h * img_in.w);
    cudaDeviceSynchronize();
    CudaErrorCheck();

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float elapsed2 = 0;
    cudaEventElapsedTime(&elapsed2, start, stop);

    total_time = (double) (elapsed2/1000) + min_time + (double) (elapsed1/1000);

    printf("Histogram equalization time = %lf sec\n", (double) (elapsed2/1000) + min_time);

    cudaEventDestroy(begin);
    cudaEventDestroy(finish);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Total GPU time: %lf sec \n", total_time);

    cudaFree(lut);
    cudaFree(hist);
    cudaFree(source_image);

    return result;
}
