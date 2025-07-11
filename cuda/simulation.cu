#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "simulation.cuh"
#include "fluid.h"

__global__ void simulateSPH(float4* pos, int count, FluidParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float x = (i % 32) * 0.06f - 1.0f;
    float y = (i / 32) * 0.06f - 1.0f;

    y += params.gravity.y * 0.01f;
    y *= (1.0f - params.viscosity * 0.05f);

    pos[i] = make_float4(x, y, 0.0f, 1.0f);
}

void simulateParticles(float4* pos, int count, FluidParams params) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    simulateSPH <<<blocks, threads>>> (pos, count, params);
}
