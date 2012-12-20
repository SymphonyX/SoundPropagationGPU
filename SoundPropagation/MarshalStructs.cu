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

	  for (int i = 0; i < 4; i++) 
	  {
		  sizeOfIn[i] = 0;
		  sizeOfOut[i] = 0;
		  for (int j = 0; j < 100; j++)
		  {
			  IN[i][j] = SoundPacketStruct(0.0f);
			  OUT[i][j] = SoundPacketStruct(0.0f);
		  }
	  }
	}

	__host__ __device__ void addPacketToIn(int direction, SoundPacketStruct packet)
	{
		int index = sizeOfIn[direction];
		IN[direction][index] = packet;
		sizeOfIn[direction] = index + 1;
	}

	__host__ __device__ void addPacketToOut(int direction, SoundPacketStruct packet)
	{
		int index = sizeOfOut[direction];
		OUT[direction][index] = packet;
		sizeOfOut[direction] = index + 1;
	}

	SoundPacketStruct IN[4][100];
	SoundPacketStruct OUT[4][100];
	int sizeOfIn[4];
	int sizeOfOut[4];
	float epsilon;
	float absorptionRate;
	float reflectionRate;
	bool flagWall;
	bool updated;
	int x;
	int z;
};

struct SoundGridToReturn {
	SoundGridToReturn(float e, float a, float r, bool w, bool u, int _x, int _z) : epsilon(e), absorptionRate(a), reflectionRate(r), flagWall(w), updated(u), x(_x), z(_z) 
	{
		for (int i = 0; i < 400; i++)
		{
			IN[i] = SoundPacketStruct(0.0f);
			OUT[i] = SoundPacketStruct(0.0f);
		}
	} 

	SoundPacketStruct IN[400];
	SoundPacketStruct OUT[400];
	int sizeOfIn[4];
	int sizeOfOut[4];
	int x;
	int z;
	float epsilon;
	float absorptionRate;
	float reflectionRate;
	bool flagWall;
	bool updated;
};

struct SoundSourceStruct {
	SoundSourceStruct(int _x, int _z) 
	{ 
		x =_x; 
		z = _z; 
		limitTickCount = 100000;

		int len = 10;
		for (int i = 0; i < 150; ++i) {
			packetList[i][0] =  SoundPacketStruct(i < len ? 10.0f : 0.1f);
			sizesOfPacketList[i] = 1;
		}
	}

	SoundPacketStruct packetList[150][10];
	int sizesOfPacketList[150];
	int limitTickCount;
	int x;
	int z;
};

struct SoundStructToReturn
{
	SoundStructToReturn(int _limitTick, int _x, int _z) : x(_x), z(_z), limitTick(_limitTick)
	{
		for (int i = 0; i < 150; i ++)
		{
			for (int j = 0; j < 10; j++)
			{
				SoundPacketStruct packet = SoundPacketStruct(0.0f);
				packetList[(i*10)+j] = packet;
			}
			sizeOfPacketList[i] = 0;
		}
	}
	SoundPacketStruct packetList[1500];
	int sizeOfPacketList[150];
	int limitTick;
	int x;
	int z;
};

#endif