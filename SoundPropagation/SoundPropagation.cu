#include "MarshalStructs.h"

extern "C" void runMainLoopKernel(int columns, int rows, SoundGridStruct* soundMap, SoundSourceStruct* soundSource) 
{
	dim3 blocks(1,1);
	dim3 threads(columns, rows);

	
	SoundGridStruct* soundMap_dev;
	cudaMalloc((void**)&soundMap_dev, (rows*columns)*sizeof(SoundGridStruct));
	cudaMemcpy(soundMap_dev, soundMap, (rows*columns)*sizeof(SoundGridStruct), cudaMemcpyHostToDevice);

	SoundSourceStruct* soundSource_dev;
	cudaMemcpy(soundSource_dev, soundSource, sizeof(SoundSourceStruct), cudaMemcpyHostToDevice);

	//cudaEmit()
	//cudaMerge()
	//cudaScatter()
	//cudaCollect()
	//cudaTick()
}