#pragma once
#include <cuda_runtime.h>
#include "fluid.h"

void simulateParticles(float4* posVBO, int count, FluidParams params, bool reset, bool paused);
