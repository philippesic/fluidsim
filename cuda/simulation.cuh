#pragma once
#include <cuda_runtime.h>
#include "fluid.h"

#define GRID_CELL_SIZE_FACTOR 1.1f

struct Grid {
    int3 gridSize;
    float cellSize;
    float3 origin;
    int* cellStart;
    int* cellEnd;
    int* particleIndices;
};

void initializeGrid(Grid& grid, const FluidParams& params, int numParticles);
void freeGrid(Grid& grid);
void assignParticlesToGrid(
    float4* positions, int* particleCellHashes, int* particleIndices, int numParticles, const Grid& grid, cudaStream_t stream);
void buildCellRanges(
    int* particleCellHashes, int* particleIndices, int numParticles, Grid& grid, cudaStream_t stream);

void simulateParticles(float4* posVBO, int count, FluidParams params, bool reset, bool paused);
