/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates how use texture fetches in CUDA
 *
 * This sample takes an input PGM image (image_filename) and generates
 * an output PGM image (image_filename_out).  This CUDA kernel performs
 * a simple 2D transform (rotation) on the texture coordinates (u,v).
 */

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "bmp/EasyBMP.h"

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check


// Define the files that are to be save and the reference images for validation
const char *imageFilename = "data/eltsin2048.bmp";
const char *cpuOutImage = "data/eltsin2048_cpu.bmp";
const char *gpuOutImage = "data/eltsin2048_gpu.bmp";


//texture<float, cudaTextureType2D, cudaReadModeElementType> tex;
const int mask_size = 3;

////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////

__global__ void medianFilter(float *output, int imageWidth, int imageHeight, cudaTextureObject_t tex) {
	//  choose element
	int col = blockIdx.x *  blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float mask[mask_size*mask_size] = { 0 };

	int k = 0;
	//fill mask with image pixels' values  
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			mask[k] = tex2D<float>(tex, col + j, row + i);
			k++;
		}
	}

	//simple sorting 
	for (int i = 1; i < mask_size*mask_size; i++) {
		for (int j = i; j > 0 && mask[j - 1] > mask[j]; j--) {
			int tmp = mask[j - 1];
			mask[j - 1] = mask[j];
			mask[j] = tmp;
		}
	}
	//write result
	output[row * imageWidth + col] = mask[4];

}


float *readImage(char *filePathInput, unsigned int *rows, unsigned int *cols) {
	BMP Image;
	Image.ReadFromFile(filePathInput);
	*rows = Image.TellHeight();
	*cols = Image.TellWidth();
	float *imageAsArray = (float *)calloc(*rows * *cols, sizeof(float));
	// Code suggested by libarary's author to read image in grayscale
	for (int i = 0; i < Image.TellWidth(); i++) {
		for (int j = 0; j < Image.TellHeight(); j++) {
			double Temp = 0.30*(Image(i, j)->Red) + 0.59*(Image(i, j)->Green) + 0.11*(Image(i, j)->Blue);
			imageAsArray[j * *cols + i] = Temp;
		}
	}
	return imageAsArray;
}

void writeImage(char *filePath, float *grayscale, unsigned int rows, unsigned int cols) {
	BMP Output;
	Output.SetSize(cols, rows);
	// set calculated value to pixel for each channel to get grayscale image 
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			RGBApixel pixel;
			pixel.Red = grayscale[i * cols + j];
			pixel.Green = grayscale[i * cols + j];
			pixel.Blue = grayscale[i * cols + j];
			pixel.Alpha = 0;
			Output.SetPixel(j, i, pixel);
		}
	}
	Output.WriteToFile(filePath);
}


float* medianCPU(float *grayscale, unsigned int rows, unsigned int cols) {
	float mask[mask_size*mask_size] = {0};
	float *image = (float *)calloc(rows * cols, sizeof(float));

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			//add padding
			if ((row == 0) || (col == 0) || (row == rows - 1) || (col == cols - 1)) {
				image[row*cols + col] = 0;
			}
			else {
				//fill mask
				for (int x = 0; x < mask_size; x++) {
					for (int y = 0; y < mask_size; y++) {
						mask[x*mask_size + y] = grayscale[(row + x - 1)*cols + (col + y - 1)];
					}
				} 
				//sort and choose middle elem
				std::sort(mask, mask + mask_size * mask_size);
				image[row*cols + col] = mask[4];

			}
		}
	}
	return image;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	int devID = findCudaDevice(argc, (const char **)argv);

	unsigned int width, height;

	float * source = readImage("data/eltsin_noise.bmp", &width, &height);

	clock_t  start_time = clock();
	float * resultCPU = medianCPU(source, width, height);
	clock_t  end_time = clock();
	std::cout << "CPU time = " << (double)((end_time - start_time) * 1000 / CLOCKS_PER_SEC) << " ms" << std::endl;
	writeImage("data/resultCPU.bmp", resultCPU, width, height);

	unsigned int size = width * height * sizeof(float);

	float *dData = NULL;
	checkCudaErrors(cudaMalloc((void **)&dData, size));

	// Allocate array and copy image data
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *cuArray;
	checkCudaErrors(cudaMallocArray(&cuArray,
		&channelDesc,
		width,
		height));
	checkCudaErrors(cudaMemcpyToArray(cuArray,
		0,
		0,
		source,
		size,
		cudaMemcpyHostToDevice));

	cudaTextureObject_t         tex;
	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = cuArray;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;

	checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

	dim3 dimBlock(16, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
		(height + dimBlock.y - 1) / dimBlock.y);


	checkCudaErrors(cudaDeviceSynchronize());
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	// Execute the kernel
	medianFilter << <dimGrid, dimBlock, 0 >> > (dData, width, height, tex);

	// Check if kernel execution generated an error
	getLastCudaError("Kernel execution failed");

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&timer);

	printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));

	// Allocate mem for the result on host side
	float *hOutputData = (float *)malloc(size);
	// copy result from device to host
	checkCudaErrors(cudaMemcpy(hOutputData,
		dData,
		size,
		cudaMemcpyDeviceToHost));

	// Write result to file
	writeImage("data/resultGPU.bmp", hOutputData, width, height);

	checkCudaErrors(cudaDestroyTextureObject(tex));
	checkCudaErrors(cudaFree(dData));
	checkCudaErrors(cudaFreeArray(cuArray));
	free(resultCPU);
	free(hOutputData);
}

