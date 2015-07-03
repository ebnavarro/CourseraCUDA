#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2
#define clamp(x) (min(max(x,0.0),1.0))

//@@ INSERT CODE HERE
#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + 4)
// implement the tiled 2D convolution kernel with adjustments for channels
// use shared memory to reduce the number of global accesses, handle the boundary conditions in when loading input list elements into the shared memory
__global__ void convolution_kernel(float *P, float *N,
									int height, int width, int channels,
									const float* __restrict__ M) {
	// Shifting from output coordinates to input coordinate
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y*O_TILE_WIDTH + ty;
	int col_o = blockIdx.x*O_TILE_WIDTH + tx;
	int row_i = row_o -2;
	int col_i = col_o -2;
	
	
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			for (int k=0; k<width; k++) {
				float accum = 0.0f;
				if(ty < O_TILE_WIDTH && tx< O_TILE_WIDTH){
					for (int y=0; y<Mask_width; y++) {
						for (int x=0; x<Mask_width; x++) {
							//accum += P[y+ty][x+tx] * M[y][x];
							accum += P[0] * M[0];
						}
					}
				}
				if(row_o < height && col_o < width){
					int index = (row_o*width) + col_o;
					N[index] = clamp(accum);
				}
			}
		}
	}
	
	
/*	// Taking Care of Boundaries (1 channel)
	if((row_i>= 0) && (row_i< height) &&
			(col_i>= 0) && (col_i< width) ) {
		Ns[ty][tx] = data[row_i*width + col_i];
	} else{
		Ns[ty][tx] = 0.0f;
	}
	
	// Some threads do not participate in calculating output.(1 channel)
	float output = 0.0f;
	if(ty < O_TILE_WIDTH && tx< O_TILE_WIDTH){
		for(i = 0; i < MASK_WIDTH; i++) {
			for(j = 0; j < MASK_WIDTH; j++) {
				output += M[i][j] * Ns[i+ty][j+tx];
			}
		}
	}
	
	// Some threads do not write output (1 channel)
	if(row_o< height && col_o< width)
		data[row_o*width + col_o] = output;
*/	
	
}


int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	// initialize thread block and kernel grid dimensions
	dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH);
	dim3 dimGrid((imageWidth-1)/O_TILE_WIDTH+1, (imageHeight-1)/O_TILE_WIDTH+1, 1);
	wbLog(TRACE, "dimGrid: ", (imageWidth-1)/O_TILE_WIDTH+1, " x ", (imageHeight-1)/O_TILE_WIDTH+1, " x ", 1);
	wbLog(TRACE, "M: ", hostInputImageData[0]);
	
	// invoke CUDA kernel
    convolution_kernel<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, 
                                       imageChannels, imageWidth, imageHeight, deviceMaskData);

	
	
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
