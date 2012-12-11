#ifndef MarshalStructs_H
#define MarshalStructs_H

struct SoundPacketStruct{
	SoundPacketStruct(float _amplitude) { amplitude = _amplitude; minRange = 0.0f; maxRange = 1.0f; }

	float amplitude;
	float minRange;
	float maxRange;
};

struct SoundGridStruct {
	SoundGridStruct(int _x, int _z) { x = _x; z = _z; epsilon = 0.001f; absorptionRate = 0.98f; reflectionRate = 0.01f;}

	float epsilon;
	float absorptionRate;
	float reflectionRate;
	int x;
	int z;
};

#endif