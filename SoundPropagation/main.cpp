
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MarshalStructs.cu"

#include <stdio.h>
#include <stdlib.h>

#define CALL __stdcall
#define EXPORT __declspec(dllexport)

extern "C" void runMainLoopKernel(int columns, int rows, SoundGridStruct* soundMap, SoundSourceStruct* soundSource, int tick, cudaDeviceProp deviceProperties, float* map);
extern "C" void cleanCuda();
extern "C" void cleanSourceCuda();

cudaDeviceProp selectedGPUProp;
SoundGridStruct* SoundMap;
SoundSourceStruct* SoundSource;
int rows; int columns;

extern "C" EXPORT void cleanMaps()
{
	cleanCuda();
}

extern "C" EXPORT void cleanSoundSource()
{
	cleanSourceCuda();
}

extern "C" EXPORT void createSoundMap(int _rows, int _columns) 
{
	free(SoundMap);
	rows = _rows;
	columns = _columns;
	size_t mapSize = (rows*columns*sizeof(SoundGridStruct));
	SoundMap = (SoundGridStruct*) malloc(mapSize);

	int i, j;
	for (i = 0; i < rows; ++i) {
		for (j = 0; j < columns; ++j) {
			SoundGridStruct newSoundGrid(j, i);
			SoundMap[i*columns+j] = newSoundGrid;
		}
	}
}

extern "C" EXPORT void selectBestGPU()
{
	int devCount;
    cudaGetDeviceCount(&devCount);

	int processorCount = 0;
	int selectedGPU = 0;
    for (int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
		if (processorCount < devProp.multiProcessorCount) {
			selectedGPU = i;
			processorCount = devProp.multiProcessorCount;
			selectedGPUProp = devProp;
		}
    }
	cudaSetDevice(selectedGPU);
}

extern "C" EXPORT void flagWall(int x, int z, float reflectionRate)
{
	SoundMap[z*columns+x].reflectionRate = reflectionRate;
	SoundMap[z*columns+x].flagWall = true;
}

extern "C" EXPORT void initSoundSource(int x, int z)
{
	free(SoundSource);
	SoundSource = (SoundSourceStruct*)malloc(sizeof(SoundSourceStruct));
	SoundSourceStruct soundSource = SoundSourceStruct(x, z);
	*SoundSource = soundSource; 
}

extern "C" EXPORT void runMainLoop(int tick, float map[])
{
	float* map_ptr = map;
	runMainLoopKernel(columns, rows, SoundMap, SoundSource, tick, selectedGPUProp, map_ptr);
}

extern "C" EXPORT float sumInForPosition(int x, int z)
{
	float value = 0.0f;
	SoundGridStruct* soundGrid = &SoundMap[z*columns+x];
	for (int direction = 0; direction < 4; direction++)
	{
		for (int i = 0; i < soundGrid->sizeOfIn[direction]; i++)
		{
			SoundPacketStruct* frame = soundGrid->IN[direction];
			value += frame[i].amplitude;
		}
	}
	return value;
}

SoundGridToReturn convertGrid(SoundGridStruct s)
{
	SoundGridToReturn g = SoundGridToReturn(s.epsilon, s.absorptionRate, s.reflectionRate, s.flagWall, s.updated, s.x, s.z);
	for (int i = 0; i < 4; i++)
	{
		g.sizeOfIn[i] = s.sizeOfIn[i];
		g.sizeOfOut[i] = s.sizeOfOut[i];
		for (int j = 0; j < s.sizeOfIn[i]; j++)
		{
			g.IN[(i*100)+j] = s.IN[i][j];
		}
		for (int j = 0; j < s.sizeOfOut[i]; j++)
		{
			g.OUT[(i*100)+j] = s.OUT[i][j];
		}

	}
	return g;
}

extern "C" EXPORT void CALL returnSoundGrid(int x, int z, SoundGridToReturn* grid)
{
	SoundGridStruct s = SoundMap[z*columns+x];
	*grid = convertGrid(s);

}

extern "C" EXPORT void CALL returnSoundSource(SoundStructToReturn* soundSourceToReturn)
{
	SoundStructToReturn s(SoundSource->limitTickCount, SoundSource->x, SoundSource->z);

	for (int i = 0; i < 150; i++)
	{
		int frameSize = SoundSource->sizesOfPacketList[i];
		s.sizeOfPacketList[i] = frameSize;
		for (int j = 0; j < frameSize; j++)
		{
			SoundPacketStruct* frame = SoundSource->packetList[i];
			SoundPacketStruct packet = *(frame+j);
			s.packetList[(i*10)+j] = packet;
		}
	}
	*soundSourceToReturn = s;
}


