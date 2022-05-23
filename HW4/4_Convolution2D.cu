/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//#define CPU_use
#define filterR 32
#define BLOCKING 512
#define padded_blocking (2 * filterR + BLOCKING) 
#define FILTER_LENGTH 	(2 * filterR + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.5

#define CudaErrorCheck() \
	error = cudaGetLastError(); \
	if (error != cudaSuccess) { \
    	printf("Cuda error %s: %d: '%s'.\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
   		freeHostandDevice(blocked_buffer1, blocked_input1, blocked_output1, blocked_buffer2, blocked_input2, blocked_output2, buffer, h_Input, h_Filter, h_OutputGPU, d_OutputGPU1, d_Buffer1, d_Input1, d_OutputGPU2, d_Buffer2, d_Input2); \
   		cudaDeviceReset(); \
   		exit(1); \
	}

__constant__ double d_Filter[FILTER_LENGTH];


void freeifdef(double *h_OutputCPU, double *h_Buffer){

  free(h_OutputCPU); 
  free(h_Buffer); 
}

void freeHostandDevice(double *blocked_buffer1, double *blocked_input1, double *blocked_output1, double *blocked_buffer2, double *blocked_input2, double *blocked_output2,double *buffer, double *h_Input, double *h_Filter, double *h_OutputGPU, double *d_OutputGPU1, double *d_Buffer1, double *d_Input1, double *d_OutputGPU2, double *d_Buffer2, double *d_Input2) {

	free(blocked_buffer1); 
	free(blocked_input1); 
    free(blocked_output1); 
    free(blocked_buffer2); 
	free(blocked_input2); 
    free(blocked_output2); 
    free(buffer); 
	free(h_Input); 
	free(h_Filter); 
	free(h_OutputGPU); 
	cudaFree(d_OutputGPU1); 
	cudaFree(d_Buffer1); 
	cudaFree(d_Input1); 
	cudaFree(d_OutputGPU2); 
	cudaFree(d_Buffer2); 
	cudaFree(d_Input2); 
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
      
      //int shared_mem_size = blockDim.x + 2*filterR; 
      
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
      
     // int shared_mem_size = blockDim.x + 2*filterR; 
      
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
    *d_Input1, *d_Input2,
    *d_Buffer1, *d_Buffer2,
    *d_OutputGPU1, *d_OutputGPU2,
    *h_OutputGPU,
    *blocked_buffer1, *blocked_buffer2,
    *blocked_input1, *blocked_input2,
    *blocked_output1, *blocked_output2,
    *buffer;

#ifdef CPU_use
    double *h_Buffer, *h_OutputCPU, max_error=0;
    struct timespec  tv1, tv2;
#endif

    long int imageW, padded_imageW, imageH, padded_imageH, shared_mem_size;
    unsigned int i,j;
    cudaError_t error;
    dim3 dimGrid, dimBlock;
    cudaEvent_t start, stop;
    cudaStream_t stream_1, stream_2;
	/*printf("Enter filter radius : ");
	scanf("%d", &filterR);*/

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%ld", &imageW);
    imageH = imageW;
    
    padded_imageH = 2*filterR + imageH;
    padded_imageW = 2*filterR + imageW;


    printf("Image Width x Height = %li x %li\n\n", imageW, imageH);
    
    
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)malloc(padded_imageW * padded_imageH * sizeof(double));
#ifdef CPU_use
    h_Buffer    = (double *)malloc(padded_imageW * padded_imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(padded_imageW * padded_imageH * sizeof(double));
#endif
    h_OutputGPU = (double *)malloc(padded_imageW * padded_imageH * sizeof(double));
    blocked_buffer1 = (double *)malloc(padded_blocking * padded_blocking * sizeof(double));
    blocked_input1 = (double *)malloc(padded_blocking * padded_blocking * sizeof(double));
    blocked_output1 = (double *)malloc(padded_blocking * padded_blocking * sizeof(double));
    blocked_buffer2 = (double *)malloc(padded_blocking * padded_blocking * sizeof(double));
    blocked_input2 = (double *)malloc(padded_blocking * padded_blocking * sizeof(double));
    blocked_output2 = (double *)malloc(padded_blocking * padded_blocking * sizeof(double));
    buffer = (double *)malloc(padded_imageW * padded_imageH * sizeof(double));

    if((h_Filter==NULL) || (h_Input==NULL) || (buffer==NULL) || (blocked_output1==NULL) || (h_OutputGPU==NULL) || (blocked_input1==NULL) || (blocked_buffer1==NULL) || (blocked_input2==NULL) || (blocked_buffer2==NULL) || (blocked_output2==NULL)){
    	printf("Error while allocating host memory.\n");
    	exit(1);
    }

#ifdef CPU_use  
    if((h_Buffer==NULL) || (h_OutputCPU==NULL)){
      printf("Error while allocating host memory.\n");
      exit(1);
    }
#endif
    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.
    
    printf("Allocating and initializing device arrays...\n");
    cudaMalloc( (void**)&d_Input1, padded_blocking * padded_blocking * sizeof(double));
    cudaMalloc( (void**)&d_Buffer1, padded_blocking * padded_blocking * sizeof(double));
    cudaMalloc( (void**)&d_OutputGPU1,padded_blocking * padded_blocking * sizeof(double));
    cudaMalloc( (void**)&d_Input2, padded_blocking * padded_blocking * sizeof(double));
    cudaMalloc( (void**)&d_Buffer2, padded_blocking * padded_blocking * sizeof(double));
    cudaMalloc( (void**)&d_OutputGPU2,padded_blocking * padded_blocking * sizeof(double));      
   
    
    if( (d_Input1==NULL) || (d_Buffer1==NULL) || (d_OutputGPU1==NULL) || (d_Input2==NULL) || (d_Buffer2==NULL) || (d_OutputGPU2==NULL) ){
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
   	
      dimGrid.x = BLOCKING / 32;
   	  dimGrid.y = BLOCKING / 32;
    
    }

    shared_mem_size = 2*filterR + dimBlock.x;
    
 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);
    
    /**************************************   CPU computation   ******************************************/   
#ifdef CPU_use
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter,padded_imageW, padded_imageH); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, padded_imageW, padded_imageH); // convolution kata sthles
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
#endif
    /**************************************   GPU computation   ******************************************/   
    
    cudaMemset(d_Buffer1, 0, padded_blocking * padded_blocking * sizeof(double));
    CudaErrorCheck();
    cudaMemset(d_Buffer2, 0, padded_blocking * padded_blocking * sizeof(double));
    CudaErrorCheck();
    cudaMemset(d_OutputGPU1, 0, padded_blocking * padded_blocking * sizeof(double));
    CudaErrorCheck();
    cudaMemset(d_OutputGPU2, 0, padded_blocking * padded_blocking * sizeof(double));
    CudaErrorCheck();

    printf("GPU computation...\n");

    cudaEventRecord(start);
    
    cudaMemcpyToSymbol( d_Filter, h_Filter, FILTER_LENGTH * sizeof(double) , 0, cudaMemcpyHostToDevice );
    CudaErrorCheck();
    //cudaMemcpy( d_Input, h_Input, padded_imageW * padded_imageH * sizeof(double) , cudaMemcpyHostToDevice );
    //CudaErrorCheck();
    

    for(int y=0; y < imageW/BLOCKING; y++){
      for(int x=0; x < imageW/BLOCKING; x=x+2){

        for(i=0; i < padded_blocking; i++){
          for(j=0; j < padded_blocking; j++){
            blocked_input1[i*padded_blocking + j] = h_Input[(i + y*BLOCKING)*padded_imageW + BLOCKING*x + j];
            blocked_input2[i*padded_blocking + j] = h_Input[(i + y*BLOCKING)*padded_imageW + BLOCKING*(x+1) + j];
          }
        }

        cudaMemcpyAsync(d_Input1, blocked_input1, padded_blocking*padded_blocking*sizeof(double), cudaMemcpyHostToDevice,stream_1);
        CudaErrorCheck();
        cudaMemcpyAsync(d_Input2, blocked_input2, padded_blocking*padded_blocking*sizeof(double), cudaMemcpyHostToDevice,stream_2);
        CudaErrorCheck();

        convolutionRowTiledGPU<<<dimGrid, dimBlock, dimBlock.y*(dimBlock.x+2*filterR)*sizeof(double), stream_1>>>(d_Buffer1, d_Input1, padded_blocking, shared_mem_size);
        CudaErrorCheck();
        convolutionRowTiledGPU<<<dimGrid, dimBlock, dimBlock.y*(dimBlock.x+2*filterR)*sizeof(double), stream_2>>>(d_Buffer2, d_Input2, padded_blocking, shared_mem_size);
        CudaErrorCheck();
       
        cudaMemcpyAsync(blocked_buffer1, d_Buffer1, padded_blocking*padded_blocking*sizeof(double), cudaMemcpyDeviceToHost,stream_1);
        CudaErrorCheck();
        cudaMemcpyAsync(blocked_buffer2, d_Buffer2, padded_blocking*padded_blocking*sizeof(double), cudaMemcpyDeviceToHost,stream_2);
        CudaErrorCheck();

        cudaStreamSynchronize(stream_1);
        cudaStreamSynchronize(stream_2);

        cudaDeviceSynchronize();

        for(i=0; i < BLOCKING; i++){
          for(j=0; j < BLOCKING; j++){
            buffer[(i + filterR + y*BLOCKING)*padded_imageW + j + filterR + x*BLOCKING ] = blocked_buffer1[(i+filterR)*padded_blocking + j +filterR];
            buffer[(i + filterR + y*BLOCKING)*padded_imageW + j + filterR + (x+1)*BLOCKING ] = blocked_buffer2[(i+filterR)*padded_blocking + j +filterR];

          }
        }
      }
    }

    for(int y=0; y < imageW/BLOCKING; y++){
      for(int x=0; x < imageW/BLOCKING; x=x+2){

        for(i=0; i < padded_blocking; i++){
          for(j=0; j < padded_blocking; j++){
            blocked_buffer1[i*padded_blocking + j] =  buffer[(y*BLOCKING + i)*padded_imageW + BLOCKING*x + j];
            blocked_buffer2[i*padded_blocking + j] =  buffer[(y*BLOCKING + i)*padded_imageW + BLOCKING*(x+1) + j];
          }
        }
 
        cudaMemcpyAsync(d_Buffer1, blocked_buffer1, padded_blocking * padded_blocking * sizeof(double), cudaMemcpyHostToDevice,stream_1);
        CudaErrorCheck();
        cudaMemcpyAsync(d_Buffer2, blocked_buffer2, padded_blocking * padded_blocking * sizeof(double), cudaMemcpyHostToDevice,stream_2);
        CudaErrorCheck();

        convolutionColumnTiledGPU<<<dimGrid, dimBlock, dimBlock.y*(dimBlock.x+2*filterR)*sizeof(double), stream_1>>>(d_OutputGPU1, d_Buffer1, padded_blocking, shared_mem_size);
        CudaErrorCheck();
        convolutionColumnTiledGPU<<<dimGrid, dimBlock, dimBlock.y*(dimBlock.x+2*filterR)*sizeof(double), stream_2>>>(d_OutputGPU2, d_Buffer2, padded_blocking, shared_mem_size);
        CudaErrorCheck();

        cudaMemcpyAsync(blocked_output1, d_OutputGPU1, padded_blocking*padded_blocking*sizeof(double), cudaMemcpyDeviceToHost,stream_1);
        CudaErrorCheck();
        cudaMemcpyAsync(blocked_output2, d_OutputGPU2, padded_blocking*padded_blocking*sizeof(double), cudaMemcpyDeviceToHost,stream_2);
        CudaErrorCheck();

        cudaStreamSynchronize(stream_1);
        cudaStreamSynchronize(stream_2);

        for(i=0; i <  BLOCKING; i++){
          for(j=0; j < BLOCKING; j++){
            h_OutputGPU[(y*BLOCKING + i + filterR)*padded_imageW + j + filterR + x*BLOCKING] = blocked_output1[(i+filterR)*padded_blocking + j +filterR];
            h_OutputGPU[(y*BLOCKING + i + filterR)*padded_imageW + j + filterR + (x+1)*BLOCKING] = blocked_output2[(i+filterR)*padded_blocking + j +filterR];
          }
        }
      }
    } 
    
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);
    
   
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    
#ifdef CPU_use
    for (i = filterR; i < (imageW + filterR); i++) {
    	for (j = filterR; j < (imageH + filterR); j++){
        if( ABS(h_OutputGPU[i*padded_imageW + j] - h_OutputCPU[i*padded_imageW + j]) > max_error) {
		      max_error = ABS(h_OutputGPU[i*padded_imageW + j] - h_OutputCPU[i*padded_imageW + j]);
    	  }
      }
    }
    
    
    printf("Max absolute error: %f.\n", max_error);
    
    printf ("Total CPU time = %.9f seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));
#endif
    printf("Total GPU time = %.9f seconds\n", elapsed / 1000);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
    // free all the allocated memory
	freeHostandDevice(blocked_buffer1, blocked_input1, blocked_output1, blocked_buffer2, blocked_input2, blocked_output2, buffer, h_Input, h_Filter, h_OutputGPU, d_OutputGPU1, d_Buffer1, d_Input1, d_OutputGPU2, d_Buffer2, d_Input2);

#ifdef CPU_use
    freeifdef(h_OutputCPU, h_Buffer); 
#endif 
    
    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
