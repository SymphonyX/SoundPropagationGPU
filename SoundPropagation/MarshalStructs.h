#ifndef MarshalStructs_H
#define MarshalStructs_H

struct SoundPacketStruct{
	SoundPacketStruct(float _amplitude) { amplitude = _amplitude; minRange = 0.0f; maxRange = 1.0f; }

	float amplitude;
	float minRange;
	float maxRange;
};

struct SoundGridStruct {
	SoundGridStruct(int _x, int _z) { x = _x; z = _z; epsilon = 0.001f; absorptionRate = 0.98f; reflectionRate = 0.01f; flagWall = false;}

	SoundPacketStruct** IN;
	SoundPacketStruct** OUT;
	float epsilon;
	float absorptionRate;
	float reflectionRate;
	bool flagWall;
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

		int len = 10;
		for (int i = 0; i < 150; ++i) {
			*(packetList+i) = (SoundPacketStruct*)malloc(sizeof(SoundPacketStruct));
			SoundPacketStruct soundPacket(i < len ? 10.0f : 0.1f);
			*(packetList+i) = &soundPacket;
		}
	}

	SoundPacketStruct** packetList;
	int limitTickCount;
	int x;
	int z;
};

#endif