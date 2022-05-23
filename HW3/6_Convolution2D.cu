/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	1


#define CudaErrorCheck() \
	error = cudaGetLastError(); \
	if (error != cudaSuccess) { \
    		printf("Cuda error %s: %d: '%s'.\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
   		freeHostandDevice(h_OutputCPU, h_Buffer, h_Input, h_Filter, h_OutputGPU, d_OutputGPU, d_Buffer, d_Input, d_Filter); \
   		cudaDeviceReset(); \
   		exit(1); \
	}




void freeHostandDevice(double *h_OutputCPU, double *h_Buffer, double *h_Input, double *h_Filter, double *h_OutputGPU, double *d_OutputGPU, double *d_Buffer, double *d_Input, double *d_Filter) {

	free(h_OutputCPU); 
	free(h_Buffer); 
	free(h_Input); 
	free(h_Filter); 
	free(h_OutputGPU); 
	cudaFree(d_OutputGPU); 
	cudaFree(d_Buffer); 
	cudaFree(d_Input); 
	cudaFree(d_Filter);
}

 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter (CPU)
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
  
  
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     
      }
      
      h_Dst[y * imageW + x] = sum;
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter (CPU)
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
      }
      
      h_Dst[y * imageW + x] = sum;
    }
  }
    
}


////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter (GPU)
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, 
                       int imageW, int imageH, int filterR) {

  int k, x, y;
                      
      double sum = 0;
      x = blockIdx.x*blockDim.x + threadIdx.x;
      y = blockIdx.y*blockDim.y + threadIdx.y;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
        }     
      }
      
      d_Dst[y * imageW + x] = sum;
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter (GPU)
////////////////////////////////////////////////////////////////////////////////
__global__  void convolutionColumnGPU(double *d_Dst, double *d_Src, double *d_Filter,
    			   int imageW, int imageH, int filterR) {

  int k, x, y;
  
      double sum = 0;
      x = blockIdx.x*blockDim.x + threadIdx.x;
      y = blockIdx.y*blockDim.y + threadIdx.y;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += d_Src[d * imageW + x] * d_Filter[filterR - k];
        }   
      }
      
      d_Dst[y * imageW + x] = sum;
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
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU,
    *h_OutputGPU,
    max_error=0;
    
    
    int imageW;
    int imageH;
    unsigned int i;
    cudaError_t error;
    dim3 dimGrid, dimBlock;
    struct timespec  tv1, tv2;
    cudaEvent_t start, stop;

	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    
    
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputGPU = (double *)malloc(imageW * imageH * sizeof(double));
    
    if( (h_Filter==NULL) || (h_Input==NULL) || (h_Buffer==NULL) || (h_OutputCPU==NULL) || (h_OutputGPU==NULL) ){
    	printf("Error while allocating host memory.\n");
    	exit(1);
    }
    

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.
    
    printf("Allocating and initializing device arrays...\n");
    cudaMalloc( (void**)&d_Filter, FILTER_LENGTH * sizeof(double));
    cudaMalloc( (void**)&d_Input, imageW * imageH * sizeof(double));
    cudaMalloc( (void**)&d_Buffer, imageW * imageH *sizeof(double));
    cudaMalloc( (void**)&d_OutputGPU,imageW * imageH * sizeof(double));    
  
    if( (d_Filter==NULL) || (d_Input==NULL) || (d_Buffer==NULL) || (d_OutputGPU==NULL) ){
    	printf("Error while allocating device memory.\n");
    	exit(1);
    }
    

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (double)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
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
    
 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
     
    
    /**************************************   CPU computation   ******************************************/   

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

    /**************************************   GPU computation   ******************************************/   
    
    
    printf("GPU computation...\n");
    
    
    cudaEventRecord(start);
    
    cudaMemcpy( d_Filter, h_Filter,FILTER_LENGTH * sizeof(double) , cudaMemcpyHostToDevice );
    CudaErrorCheck();
    cudaMemcpy( d_Input, h_Input,imageW * imageH * sizeof(double) , cudaMemcpyHostToDevice );
    CudaErrorCheck();
    
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    CudaErrorCheck();
    
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    CudaErrorCheck();
    
    cudaMemcpy( h_OutputGPU, d_OutputGPU, imageW * imageH * sizeof(double) , cudaMemcpyDeviceToHost );
    CudaErrorCheck(); 
    
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);
    
   
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    
    for(i=0; i < imageW * imageH; i++){
    	if( ABS(h_OutputGPU[i] - h_OutputCPU[i]) > max_error) {
		max_error = ABS(h_OutputGPU[i] - h_OutputCPU[i]);
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
    freeHostandDevice(h_OutputCPU, h_Buffer, h_Input, h_Filter, h_OutputGPU, d_OutputGPU, d_Buffer, d_Input, d_Filter);
    
    
    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
