/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define filterR 32
#define FILTER_LENGTH 	(2 * filterR + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.5

#define CudaErrorCheck() \
	error = cudaGetLastError(); \
	if (error != cudaSuccess) { \
    		printf("Cuda error %s: %d: '%s'.\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
   		freeHostandDevice(h_OutputCPU, h_Buffer, h_Input, h_Filter, h_OutputGPU, d_OutputGPU, d_Buffer, d_Input); \
   		cudaDeviceReset(); \
   		exit(1); \
	}

__constant__ double d_Filter[FILTER_LENGTH];


void freeHostandDevice(double *h_OutputCPU, double *h_Buffer, double *h_Input, double *h_Filter, double *h_OutputGPU, double *d_OutputGPU, double *d_Buffer, double *d_Input) {

	free(h_OutputCPU); 
	free(h_Buffer); 
	free(h_Input); 
	free(h_Filter); 
	free(h_OutputGPU); 
	cudaFree(d_OutputGPU); 
	cudaFree(d_Buffer); 
	cudaFree(d_Input); 
}

 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter (CPU)
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, int padded_imageH, int padded_imageW) {

  int x, y, k;
                      
  for (y = filterR; y < (padded_imageH - filterR); y++) {
    for (x = filterR; x < (padded_imageW - filterR); x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        sum += h_Src[y * padded_imageW + x + k] * h_Filter[filterR - k];       
      }
      
      h_Dst[y * padded_imageW + x] = sum;
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter (CPU)
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter, int padded_imageH, int padded_imageW) {

  int x, y, k;
  
  for (y = filterR; y < (padded_imageH - filterR); y++) {
    for (x = filterR; x < (padded_imageW - filterR); x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        sum += h_Src[(y + k) * padded_imageW + x] * h_Filter[filterR - k];
      }

      h_Dst[y * padded_imageW + x] = sum;
    }
  }
    
}


////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter (GPU)
////////////////////////////////////////////////////////////////////////////////


__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, int imageW) {

  int k, x, y;
                      
      double sum = 0;
      x = blockIdx.x*blockDim.x + threadIdx.x + filterR;
      y = blockIdx.y*blockDim.y + threadIdx.y + filterR;

      for (k = -filterR; k <= filterR; k++) {
        sum += d_Src[y * imageW + x + k] * d_Filter[filterR - k];     
      }
      
      d_Dst[y * imageW + x] = sum;
}



__global__ void convolutionRowTiledGPU(double *d_Dst, double *d_Src, int padded_imageW, int shared_mem_size) {


      int k, x, y, tx, ty;
                      
      double sum = 0;
      x = blockIdx.x*blockDim.x + threadIdx.x + filterR;
      y = blockIdx.y*blockDim.y + threadIdx.y + filterR;
      tx = threadIdx.x;
      ty = threadIdx.y;
      extern __shared__ double s_Src[];
      
      
      for(int p=0; p < shared_mem_size; p+=blockDim.x){
      	s_Src[ty*shared_mem_size + tx + p] = d_Src[y*padded_imageW + p + x-filterR];
      }
      __syncthreads();

      for (k = -filterR; k <= filterR; k++) {
        sum += s_Src[ty*shared_mem_size + tx + k + filterR] * d_Filter[filterR - k];     
      }
      d_Dst[y * padded_imageW + x] = sum;
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter (GPU)
////////////////////////////////////////////////////////////////////////////////


__global__  void convolutionColumnGPU(double *d_Dst, double *d_Src, int imageW) {

  int k, x, y;
  
      double sum = 0;
      x = blockIdx.x*blockDim.x + threadIdx.x + filterR;
      y = blockIdx.y*blockDim.y + threadIdx.y + filterR;

      for (k = -filterR; k <= filterR; k++) {
        sum += d_Src[(y+k) * imageW + x] * d_Filter[filterR - k];  
      }
      
      d_Dst[y * imageW + x] = sum;
}

__global__  void convolutionColumnTiledGPU(double *d_Dst, double *d_Src, int padded_imageW, int shared_mem_size) {
      
      int k, x, y, tx, ty;
                      
      double sum = 0;
      x = blockIdx.x*blockDim.x + threadIdx.x + filterR;
      y = blockIdx.y*blockDim.y + threadIdx.y + filterR;
      tx = threadIdx.x;
      ty = threadIdx.y;
      extern __shared__ double s_Src[];
      
      
      for(int p=0; p < shared_mem_size; p+=blockDim.x){
      	s_Src[(ty + p)*blockDim.x + tx] = d_Src[(y-filterR + p)*padded_imageW + x];
      }
      __syncthreads();
      
      for (k = -filterR; k <= filterR; k++) {
        sum += s_Src[(ty + k + filterR)*blockDim.x + tx] * d_Filter[filterR - k];  
      }
      d_Dst[y * padded_imageW + x] = sum;
}
                      
  
     

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    double
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU,
    *h_OutputGPU,
    max_error;
    
    int imageW, padded_imageW, imageH, padded_imageH, shared_mem_size,tile_size;
    unsigned int i,j;
    cudaError_t error;
    dim3 dimGrid, dimBlock;
    struct timespec  tv1, tv2;
    cudaEvent_t start, stop;

	/*printf("Enter filter radius : ");
	scanf("%d", &filterR);*/

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;
    
    padded_imageH = 2*filterR + imageH;
    padded_imageW = 2*filterR + imageW;


    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    
    
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)malloc(padded_imageW * padded_imageH * sizeof(double));
    h_Buffer    = (double *)malloc(padded_imageW * padded_imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(padded_imageW * padded_imageH * sizeof(double));
    h_OutputGPU = (double *)malloc(padded_imageW * padded_imageH * sizeof(double));
    
    if( (h_Filter==NULL) || (h_Input==NULL) || (h_Buffer==NULL) || (h_OutputCPU==NULL) || (h_OutputGPU==NULL) ){
    	printf("Error while allocating host memory.\n");
    	exit(1);
    }
    

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.
    
    printf("Allocating and initializing device arrays...\n");
    cudaMalloc( (void**)&d_Input, padded_imageW * padded_imageH * sizeof(double));
    cudaMalloc( (void**)&d_Buffer, padded_imageW * padded_imageH * sizeof(double));
    cudaMalloc( (void**)&d_OutputGPU,padded_imageW * padded_imageH * sizeof(double));    
  
    if( (d_Input==NULL) || (d_Buffer==NULL) || (d_OutputGPU==NULL) ){
    	printf("Error while allocating device memory.\n");
    	exit(1);
    }
    

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (double)(rand() % 16);
    }

    for (i = 0; i < padded_imageW; i++) {
    	for (j = 0; j < padded_imageH; j++){
            h_Input[i*padded_imageW+j] = 0;
        }
    }

    for (i = filterR; i < (padded_imageW - filterR); i++) {
    	for (j = filterR; j < (padded_imageH - filterR); j++){
            h_Input[i*padded_imageW+j] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
        }
    }
    
    
    if(imageW <= 32){
    	dimBlock.x = imageW;
   	dimBlock.y = imageH;
   	
   	dimGrid.x = 1;
   	dimGrid.y = 1;
    }
    
    else{
    
   	dimBlock.x = 32;
   	dimBlock.y = 32;
   	
   	dimGrid.x = imageW / 32;
   	dimGrid.y = imageH / 32;
    
    }
    
 	shared_mem_size = 2*filterR + dimBlock.x;
 	tile_size = dimBlock.y*(dimBlock.x+2*filterR)*sizeof(double);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
     
    
    /**************************************   CPU computation   ******************************************/   

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter,padded_imageW, padded_imageH); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, padded_imageW, padded_imageH); // convolution kata sthles
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    /**************************************   GPU computation   ******************************************/   
    
    
    printf("GPU computation...\n");
    
    
    cudaEventRecord(start);
    
    cudaMemset(d_Buffer, 0, padded_imageW * padded_imageW * sizeof(double));
    CudaErrorCheck();
    cudaMemcpyToSymbol( d_Filter, h_Filter, FILTER_LENGTH * sizeof(double) , 0, cudaMemcpyHostToDevice );
    CudaErrorCheck();
    cudaMemcpy( d_Input, h_Input, padded_imageW * padded_imageH * sizeof(double) , cudaMemcpyHostToDevice );
    CudaErrorCheck();
    
    convolutionRowTiledGPU<<<dimGrid, dimBlock, tile_size>>>(d_Buffer, d_Input, padded_imageW, shared_mem_size);
    cudaDeviceSynchronize();
    CudaErrorCheck();
    
    convolutionColumnTiledGPU<<<dimGrid, dimBlock, tile_size>>>(d_OutputGPU, d_Buffer, padded_imageW, shared_mem_size);
    cudaDeviceSynchronize();
    CudaErrorCheck();
    
    cudaMemcpy( h_OutputGPU, d_OutputGPU, padded_imageW * padded_imageH * sizeof(double) , cudaMemcpyDeviceToHost );
    CudaErrorCheck(); 
    
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);
    
   
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    
    
    for (i = filterR; i < (padded_imageW - filterR); i++) {
    	for (j = filterR; j < (padded_imageH - filterR); j++){
            if( ABS(h_OutputGPU[i*padded_imageW + j] - h_OutputCPU[i*padded_imageW + j]) > max_error) {
		max_error = ABS(h_OutputGPU[i*padded_imageW + j] - h_OutputCPU[i*padded_imageW + j]);
    	}
        }
    }
    
    
    printf("Max absolute error: %f.\n", max_error);
    
    printf ("Total CPU time = %.9f seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

    printf("Total GPU time = %.9f seconds\n", elapsed / 1000);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // free all the allocated memory
    freeHostandDevice(h_OutputCPU, h_Buffer, h_Input, h_Filter, h_OutputGPU, d_OutputGPU, d_Buffer, d_Input);
    
    
    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
