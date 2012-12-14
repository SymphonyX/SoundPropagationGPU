#ifndef MarshalStructs_H
#define MarshalStructs_H
#include <cuda_runtime.h>

class SoundPacketStruct{

	public:
	__host__ __device__ SoundPacketStruct(float _amplitude): amplitude(_amplitude), minRange(0.0f), maxRange(1.0f) { }
	__host__ __device__ SoundPacketStruct(float _amplitude, float _minRange, float _maxRange) : amplitude(_amplitude), minRange(_minRange), maxRange(_maxRange) { }
	__host__ __device__ SoundPacketStruct() : amplitude(0.0f), minRange(0.0f), maxRange(1.0f) { }
	float amplitude;
	float minRange;
	float maxRange;
};

struct SoundGridStruct {
	SoundGridStruct(int _x, int _z) 
	{ x = _x; 
	  z = _z; 
	  epsilon = 0.001f; 
	  absorptionRate = 0.98f; 
	  reflectionRate = 0.01f; 
	  flagWall = false;
	  updated = false;

	  IN = (SoundPacketStruct**)malloc(4*sizeof(SoundPacketStruct*));
	  OUT = (SoundPacketStruct**)malloc(4*sizeof(SoundPacketStruct*));
	  sizeOfIn = (int*)malloc(4*sizeof(int));
	  sizeOfOut = (int*)malloc(4*sizeof(int));
	  for (int i = 0; i < 4; i++) 
	  {
		  *(IN+i) = (SoundPacketStruct*)malloc(100*sizeof(SoundPacketStruct));
		  *(OUT+i) = (SoundPacketStruct*)malloc(100*sizeof(SoundPacketStruct));
		  *(sizeOfIn+i) = 0;
		  *(sizeOfOut+i) = 0;
	  }
	}

	SoundPacketStruct** IN;
	SoundPacketStruct** OUT;
	int* sizeOfIn;
	int* sizeOfOut;
	float epsilon;
	float absorptionRate;
	float reflectionRate;
	bool flagWall;
	bool updated;
	int x;
	int z;
};

struct SoundSourceStruct {
	SoundSourceStruct(int _x, int _z) 
	{ 
		x =_x; 
		z = _z; 
		limitTickCount = 100000;
		packetList = (SoundPacketStruct**)malloc(4 *sizeof(float*));
		sizesOfPacketList = (int*)malloc(150*sizeof(int));

		int len = 10;
		for (int i = 0; i < 150; ++i) {
			*(packetList+i) = (SoundPacketStruct*)malloc(sizeof(SoundPacketStruct));
			SoundPacketStruct soundPacket = SoundPacketStruct(i < len ? 10.0f : 0.1f);
			*(packetList+i) = &soundPacket;

			*(sizesOfPacketList+i) = 1;
		}
	}

	SoundPacketStruct** packetList;
	int* sizesOfPacketList;
	int limitTickCount;
	int x;
	int z;
};

#endif