#include <stdio.h>
#include <stdlib.h>
#include "MarshalStructs.h"

#define CALL __stdcall
#define EXPORT __declspec(dllexport)

extern "C" void runMainLoopKernel(int columns, int rows);

SoundGridStruct* SoundMap;
SoundSourceStruct* SoundSource;
int rows; int columns;

extern "C" EXPORT void createSoundMap(int _rows, int _columns) 
{
	rows = _rows;
	columns = _columns;
	size_t mapSize = (rows*sizeof(SoundGridStruct))*(columns*sizeof(SoundGridStruct));
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
	SoundSource = &soundSource; 
}