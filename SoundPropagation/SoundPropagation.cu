#include "MarshalStructs.h"

#define NUMBER_OF_DIRECTIONS 4

__global__ void emitKernel(SoundSourceStruct* soundSource, SoundGridStruct* soundMap, int rows, int columns, int tick)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (soundSource->x == x && soundSource->z == y)
	{
		SoundGridStruct* soundGrid = &soundMap[y*columns+x];
		SoundPacketStruct* frame = soundSource->packetList[tick];
		int frameSize = soundSource->sizesOfPacketList[tick];
		for (int index = 0; index < frameSize; index++) 
		{
			for (int direction = 0; direction < NUMBER_OF_DIRECTIONS; direction++)
			{
				SoundPacketStruct soundPacket(frame[index]);
				soundGrid->IN[direction] = &soundPacket;
			}
		}

	}
}

extern "C" void runMainLoopKernel(int columns, int rows, SoundGridStruct* soundMap, SoundSourceStruct* soundSource, int tick) 
{
	dim3 blocks(1,1);
	dim3 threads(columns, rows);

	
	SoundGridStruct* soundMap_dev;
	cudaMalloc((void**)&soundMap_dev, (rows*columns)*sizeof(SoundGridStruct));
	cudaMemcpy(soundMap_dev, soundMap, (rows*columns)*sizeof(SoundGridStruct), cudaMemcpyHostToDevice);

	SoundSourceStruct* soundSource_dev;
	cudaMalloc((void**)&soundSource_dev, sizeof(SoundSourceStruct));
	cudaMemcpy(soundSource_dev, soundSource, sizeof(SoundSourceStruct), cudaMemcpyHostToDevice);

	emitKernel<<<blocks, threads>>> (soundSource_dev, soundMap_dev, rows, columns, tick);
	//cudaEmit()
	//cudaMerge()
	//cudaScatter()
	//cudaCollect()
	//cudaTick()
}