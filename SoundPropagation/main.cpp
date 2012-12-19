
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MarshalStructs.cu"

#include <stdio.h>
#include <stdlib.h>

#define CALL __stdcall
#define EXPORT __declspec(dllexport)

extern "C" void runMainLoopKernel(int columns, int rows, SoundGridStruct* soundMap, SoundSourceStruct* soundSource, int tick);

SoundGridStruct* SoundMap;
SoundSourceStruct* SoundSource;
int rows; int columns;

extern "C" EXPORT void createSoundMap(int _rows, int _columns) 
{
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

extern "C" EXPORT void flagWall(int x, int z, float reflectionRate)
{
	SoundMap[z*columns+x].reflectionRate = reflectionRate;
	SoundMap[z*columns+x].flagWall = true;
}

extern "C" EXPORT void initSoundSource(int x, int z)
{
	SoundSource = (SoundSourceStruct*)malloc(sizeof(SoundSourceStruct));
	SoundSourceStruct soundSource = SoundSourceStruct(x, z);
	*SoundSource = soundSource; 
}

extern "C" EXPORT void runMainLoop(int tick)
{
	runMainLoopKernel(columns, rows, SoundMap, SoundSource, tick);
}

extern "C" EXPORT float sumInForPosition(int x, int z)
{
	float value = 0.0f;
	SoundGridStruct* soundGrid = &SoundMap[z*columns+x];
	for (int direction = 0; direction < 4; direction++)
	{
		for (int i = 0; soundGrid->sizeOfIn[direction]; i++)
		{
			SoundPacketStruct* frame = soundGrid->IN[direction];
			value += frame[i].amplitude;
		}
	}
	return value;
}

extern "C" EXPORT void CALL returnSoundGrid(int x, int z, SoundGridToReturn* grid)
{
	SoundGridStruct s = SoundMap[z*columns+x];
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
			g.IN[(i*100)+j] = s.OUT[i][j];
		}

	}
	*grid = g;

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


