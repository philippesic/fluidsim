#pragma once
#include <cuda_runtime.h>
#include "fluid.h"

void simulateParticles(float4* pos, int count, FluidParams params);
