#include <stdio.h>
#include <stdlib.h>
#include "MarshalStructs.h"

#define CALL __stdcall
#define EXPORT __declspec(dllexport)

SoundGridStruct* SoundMap;
int rows, columns;

extern "C" EXPORT void createSoundMap(int _rows, int _columns) 
{
	rows = _rows;
	columns = _columns;
	size_t mapSize = (rows*sizeof(SoundGridStruct))*(columns*sizeof(SoundGridStruct));
	SoundMap = (SoundGridStruct*) malloc(mapSize);
	int i, j;
	for (i = 0; i < rows; ++i) {
		for (j = 0; j < columns; ++j) {
			SoundMap[i*columns+j] = SoundGridStruct(j, i);
		}
	}
}