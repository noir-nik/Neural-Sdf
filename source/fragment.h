#pragma once
#include "mlp.h"

#include "LiteMath.h"
using namespace LiteMath;

extern MLP* model_frag;

float4 fragment(float2 gl_FragCoord, float2 iResolution);
void renderCPU(MLP* mlp, int width, int height, std::vector<uint32_t>& pixelData);
void renderGPU(MLP* mlp, int width, int height, std::vector<uint32_t>& pixelData);
inline uint32_t RealColorToUint32(float4 real_color)
{
	float r = real_color[0] * 256.0f;
	float g = real_color[1] * 256.0f;
	float b = real_color[2] * 256.0f;
	float a = real_color[3] * 256.0f;

	uint32_t red = max(0, min(255, (int)r));
	uint32_t green = max(0, min(255, (int)g));
	uint32_t blue = max(0, min(255, (int)b));
	uint32_t alpha = max(0, min(255, (int)a));

	return red | (green << 8) | (blue << 16) | (alpha << 24);
}