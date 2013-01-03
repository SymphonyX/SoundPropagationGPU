#include "MarshalStructs.cu"
#include <math.h>

#define NUMBER_OF_DIRECTIONS 4

enum {North = 0, East = 1, South = 2, West = 3, None = -1} Direction;
float* map_dev = NULL; 

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

__device__ SoundGridStruct* neighborAtDirection(SoundGridStruct* soundMap, SoundGridStruct* soundGrid, int direction, int rows, int columns)
{
	if (direction == North && soundGrid->z > 0) {return &soundMap[(soundGrid->z-1)*columns+soundGrid->x]; }
	else if (direction == East && soundGrid->x < columns-1) {return &soundMap[soundGrid->z*columns+(soundGrid->x+1)]; }
	else if (direction == West && soundGrid->x > 0) {return &soundMap[soundGrid->z*columns+(soundGrid->x-1)]; }
	else if (direction == South && soundGrid->z < rows-1) {return &soundMap[(soundGrid->z+1)*columns+soundGrid->x]; }
	else return NULL;
}

//************KERNELS******************//

void emitKernel(SoundSourceStruct* soundSource, SoundGridStruct* soundMap, int rows, int columns, int tick, int y, int x)
{

	tick = tick % 150;

	SoundGridStruct* soundGrid = &soundMap[y*columns+x];
	int frameSize = soundSource->sizesOfPacketList[tick];
	for (int index = 0; index < frameSize; index++) 
	{
		for (int direction = 0; direction < NUMBER_OF_DIRECTIONS; direction++)
		{
			soundGrid->addPacketToIn(direction, soundSource->packetList[tick][index]);
		}
	}
}

__global__ void mergeKernel(SoundGridStruct* soundMap, int rows, int columns)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	if (x < columns && y < rows) {
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
				soundGrid->addPacketToIn(direction, soundPacket);
			}
			soundGrid->updated = true;
		}
	}
}

__global__ void scatterKernel(SoundGridStruct* soundMap, int rows, int columns)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	
	if (x < columns && y < rows) {
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
				for (int index = 0; index < soundGrid->sizeOfIn[direction]; index++)
				{
					SoundPacketStruct* packet = &inVector[index];
					if (abs(packet->amplitude) > soundGrid->epsilon)
					{
						SoundPacketStruct packetCopy = SoundPacketStruct(packet->amplitude * soundGrid->reflectionRate);
						packetCopy.maxRange = packet->maxRange;
						packetCopy.minRange = packet->minRange;
						soundGrid->addPacketToOut(direction, packetCopy);
						soundGrid->updated = true;
					}

				}
			} 
			else 
			{

				for (int index = 0; index < soundGrid->sizeOfIn[direction]; index++)
				{
					SoundPacketStruct* soundPacket = (inVector+index);
					if (abs(soundPacket->amplitude) > soundGrid->epsilon)
					{
						SoundPacketStruct fwdPacket = SoundPacketStruct(soundGrid->absorptionRate * soundPacket->amplitude / 2, soundPacket->minRange, soundPacket->maxRange);
						soundGrid->addPacketToOut(reverseDirection(direction), fwdPacket);

						SoundPacketStruct bckPacket = SoundPacketStruct(soundGrid->absorptionRate * -soundPacket->amplitude / 2, soundPacket->minRange, soundPacket->maxRange);
						soundGrid->addPacketToOut(direction, bckPacket);

						SoundPacketStruct clockwisePacket = SoundPacketStruct(soundGrid->absorptionRate * soundPacket->amplitude / 2, soundPacket->minRange, soundPacket->maxRange);
						soundGrid->addPacketToOut(clockwiseDirection(direction), clockwisePacket);

						SoundPacketStruct ctClockwisePacket = SoundPacketStruct(soundGrid->absorptionRate * soundPacket->amplitude / 2, soundPacket->minRange, soundPacket->maxRange);
						soundGrid->addPacketToOut(counterClockwiseDirection(direction), ctClockwisePacket);

						soundGrid->updated = true;
					}
				}
			}
		}

		for (int direction = 0; direction < NUMBER_OF_DIRECTIONS; direction++) 
		{
			soundGrid->sizeOfIn[direction] = 0;
		}
	}
}

__global__ void collectKernel(SoundGridStruct* soundMap, int rows, int columns)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	
	if (x < columns && y < rows) {
		SoundGridStruct* soundGrid = &soundMap[y*columns+x];
	
		for (int direction = 0; direction < NUMBER_OF_DIRECTIONS; direction++)
		{
			int revDirection = reverseDirection(direction);
			SoundPacketStruct* frame = soundGrid->OUT[direction];
			SoundGridStruct* neighbor = neighborAtDirection(soundMap, soundGrid, direction, rows, columns);
			SoundPacketStruct* neighborFrame = NULL;
			if (neighbor != NULL)
			{
				neighborFrame = neighbor->IN[revDirection];
			}

			if (neighborFrame != NULL)
			{
				neighbor->sizeOfIn[revDirection] = 0;
				for (int index = 0; index < soundGrid->sizeOfOut[direction]; index++)
				{
					SoundPacketStruct packet = SoundPacketStruct(frame[index].amplitude, frame[index].minRange, frame[index].maxRange); 
					neighbor->addPacketToIn(revDirection, packet);
				}
			}
		}
	}
}

__global__ void computeValuesKernel(int rows, int columns, float* map, SoundGridStruct* soundMap)
{
	int x = threadIdx.x;
	int y = blockIdx.x;

	SoundGridStruct* soundGrid = &soundMap[y*columns+x];
	float value = 0.0f;
	for (int direction = 0; direction < 4; direction++)
	{
		for (int i = 0; i < soundGrid->sizeOfIn[direction]; i++)
		{
			SoundPacketStruct* frame = soundGrid->IN[direction];
			value += frame[i].amplitude;
		}
	}
	map[y*columns+x] = value;
}

extern "C" void runMainLoopKernel(int columns, int rows, SoundGridStruct* soundMap, SoundSourceStruct* soundSource, int tick, cudaDeviceProp deviceProperties, float* map) 
{
	int warpSize = deviceProperties.warpSize;
	int maxThreads = deviceProperties.maxThreadsPerBlock;
	int numberOfCores = deviceProperties.multiProcessorCount;
	int gridSize = columns*rows;

	int numberOfwarps = ceil((double)gridSize/warpSize);
	int numberOfBlocks = ceil((double)numberOfwarps/numberOfCores); 
	int numberOfThreads = ceil((double)gridSize/numberOfBlocks);
	
	dim3 blocks(rows, 1, 1);
	dim3 threads(columns, 1, 1);

	emitKernel(soundSource, soundMap, rows, columns, tick, soundSource->z, soundSource->x);

	SoundGridStruct* soundMap_dev;
	cudaMalloc((void**)&soundMap_dev, (rows*columns)*sizeof(SoundGridStruct));
	cudaMemcpy(soundMap_dev, soundMap, (rows*columns)*sizeof(SoundGridStruct), cudaMemcpyHostToDevice);

	mergeKernel<<<blocks, threads>>> (soundMap_dev, rows, columns);
	scatterKernel<<<blocks, threads>>> (soundMap_dev, rows, columns);
	collectKernel<<<blocks, threads>>> (soundMap_dev, rows, columns);

	//*****Values****//
	if (map_dev == NULL) {
		cudaMalloc((void**)&map_dev, (rows*columns)*sizeof(float));
	}

	computeValuesKernel<<<blocks, threads>>> (rows, columns, map_dev, soundMap_dev);

	cudaMemcpy(map, map_dev, (rows*columns)*sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(soundMap, soundMap_dev, (rows*columns)*sizeof(SoundGridStruct), cudaMemcpyDeviceToHost);

	cudaFree(soundMap_dev);
}



