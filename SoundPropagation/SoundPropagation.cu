#include "MarshalStructs.cu"
#include <math.h>

#define NUMBER_OF_DIRECTIONS 4

enum {North = 0, East = 1, South = 2, West = 3, None = -1} Direction;

//***********HELPER FUNCTIONS*************//

__device__ int reverseDirection(int direction)
{
	if (direction == West) return East;
	else if (direction == East) return West;
	else if (direction == North) return South;
	else if (direction == South) return North;
	else return None;
}

__device__ int clockwiseDirection(int direction)
{
	if (direction == West) return North;
	else if (direction == East) return South;
	else if (direction == North) return East;
	else if (direction == South) return West;
	else return None;
}

__device__ int counterClockwiseDirection(int direction)
{
	if (direction == West) return South;
	else if (direction == East) return North;
	else if (direction == North) return West;
	else if (direction == South) return East;
	else return None;
}

//************KERNELS******************//

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
				SoundPacketStruct soundPacket = SoundPacketStruct(0.0f);
				soundPacket.amplitude = frame[index].amplitude;
				soundPacket.minRange = frame[index].minRange;
				soundPacket.maxRange = frame[index].maxRange;
				soundGrid->IN[direction] = &soundPacket;
			}
		}

	}
}

__global__ void mergeKernel(SoundGridStruct* soundMap, int rows, int columns)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	SoundGridStruct* soundGrid = &soundMap[y*columns+x];
	soundGrid->updated = false;
	for (int direction = 0; direction < NUMBER_OF_DIRECTIONS; direction++)
	{
		SoundPacketStruct soundPacket = SoundPacketStruct(0.0f);
		for (int i = 0; i < soundGrid->sizeOfIn[direction]; i++)
		{
			SoundPacketStruct* packetListPtr = soundGrid->IN[direction]; 
			soundPacket.amplitude += (packetListPtr+i)->amplitude;
		}
		soundGrid->sizeOfIn[direction] = 0;

		if (abs(soundPacket.amplitude) > soundGrid->epsilon)
		{
			SoundPacketStruct* packetListPtr = soundGrid->IN[direction];
			*(packetListPtr+soundGrid->sizeOfIn[direction]) = soundPacket;
		}
		soundGrid->updated = true;
	}
}

__global__ void scatterKernel(SoundGridStruct* soundMap, int rows, int columns)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	SoundGridStruct* soundGrid = &soundMap[y*columns+x];
	soundGrid->updated = false;
	
	for (int direction = 0; direction < NUMBER_OF_DIRECTIONS; direction++)
	{
		soundGrid->sizeOfOut[direction] = 0;
	}

	for (int direction = 0; direction < NUMBER_OF_DIRECTIONS; direction++)
	{
		SoundPacketStruct* inVector = soundGrid->IN[direction];
		if (soundGrid->flagWall)
		{
			SoundPacketStruct* outVector = soundGrid->OUT[direction];
			for (int index = 0; index < soundGrid->sizeOfIn[direction]; index++)
			{
				SoundPacketStruct* packet = &inVector[index];
				if (abs(packet->amplitude > soundGrid->epsilon))
				{
					SoundPacketStruct packetCopy = SoundPacketStruct(packet->amplitude * soundGrid->reflectionRate);
					packetCopy.maxRange = packet->maxRange;
					packetCopy.minRange = packet->minRange;
					*(outVector+soundGrid->sizeOfOut[direction]) = packetCopy;
					soundGrid->sizeOfOut[direction] += 1;
					soundGrid->updated = true;
				}

			}
		} 
		else 
		{
			SoundPacketStruct* forwardVector = soundGrid->OUT[reverseDirection(direction)];
			SoundPacketStruct* backwardVector = soundGrid->OUT[direction];
			SoundPacketStruct* clockwiseVector = soundGrid->OUT[clockwiseDirection(direction)];
			SoundPacketStruct* counterClockwiseVector = soundGrid->OUT[counterClockwiseDirection(direction)];

			for (int index = 0; index < soundGrid->sizeOfIn[direction]; index++)
			{
				SoundPacketStruct* soundPacket = (inVector+index);
				if (abs(soundPacket->amplitude > soundGrid->epsilon))
				{
					int* fwdVectorSize = &soundGrid->sizeOfOut[reverseDirection(direction)];
					*(forwardVector+*fwdVectorSize) = SoundPacketStruct(soundGrid->absorptionRate * soundPacket->amplitude / 2, soundPacket->minRange, soundPacket->maxRange);
					*fwdVectorSize +=1;

					int* backwardVectorSize = &soundGrid->sizeOfOut[direction];
					*(backwardVector+*backwardVectorSize) = SoundPacketStruct(soundGrid->absorptionRate * -soundPacket->amplitude / 2, soundPacket->minRange, soundPacket->maxRange);
					*backwardVectorSize +=1;

					int* clockwiseVectorSize = &soundGrid->sizeOfOut[clockwiseDirection(direction)];
					*(clockwiseVector+*clockwiseVectorSize) = SoundPacketStruct(soundGrid->absorptionRate * soundPacket->amplitude / 2, soundPacket->minRange, soundPacket->maxRange);
					*clockwiseVectorSize +=1;

					int* counterClockwiseVectorSize = &soundGrid->sizeOfOut[counterClockwiseDirection(direction)];
					*(counterClockwiseVector+*counterClockwiseVectorSize) = SoundPacketStruct(soundGrid->absorptionRate * soundPacket->amplitude / 2, soundPacket->minRange, soundPacket->maxRange);
					*counterClockwiseVectorSize +=1;

					soundGrid->updated = true;
				}
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
	mergeKernel<<<blocks, threads>>> (soundMap_dev, rows, columns);
	scatterKernel<<<blocks, threads>>> (soundMap_dev, rows, columns);
	//cudaEmit()
	//cudaMerge()
	//cudaScatter()
	//cudaCollect()
	//cudaTick()
}
