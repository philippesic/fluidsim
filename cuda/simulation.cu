#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "simulation.cuh"
#include "fluid.h"
#include "utils.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <iostream>

__device__ __forceinline__ int dmin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int dmax(int a, int b) { return a > b ? a : b; }

struct Particle {
    float3 position;
    float3 velocity;
    float density;
    float pressure;
};

// SPH constants
__device__ float poly6(float r2, float h) {
    float diff = h * h - r2;
    return (315.0f / (64.0f * M_PI * pow(h, 9))) * pow(diff, 3);
}

__device__ float3 spikyGrad(float3 r, float h) {
    float rLen = length(r);
    if (rLen == 0.0f) return make_float3(0,0,0);
    float coeff = -45.0f / (M_PI * pow(h, 6)) * pow(h - rLen, 2);
    return coeff * (r / rLen);
}

__device__ int getCellHash(const float3& pos, const Grid& grid) {
    int3 cell;
    cell.x = int((pos.x - grid.origin.x) / grid.cellSize);
    cell.y = int((pos.y - grid.origin.y) / grid.cellSize);
    cell.z = int((pos.z - grid.origin.z) / grid.cellSize);
    cell.x = dmax(0, dmin(cell.x, grid.gridSize.x - 1));
    cell.y = dmax(0, dmin(cell.y, grid.gridSize.y - 1));
    cell.z = dmax(0, dmin(cell.z, grid.gridSize.z - 1));
    return (cell.z * grid.gridSize.y * grid.gridSize.x) + (cell.y * grid.gridSize.x) + cell.x;
}

__device__ void forEachNeighborCell(
    int3 cell, const Grid& grid,
    void (*func)(int, void*), void* userData)
{
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cell.x + dx;
                int ny = cell.y + dy;
                int nz = cell.z + dz;
                if (nx < 0 || ny < 0 || nz < 0 ||
                    nx >= grid.gridSize.x ||
                    ny >= grid.gridSize.y ||
                    nz >= grid.gridSize.z)
                    continue;
                int neighborHash = (nz * grid.gridSize.y * grid.gridSize.x) + (ny * grid.gridSize.x) + nx;
                func(neighborHash, userData);
            }
}

__device__ int3 getCellCoord(const float3& pos, const Grid& grid) {
    int3 cell;
    cell.x = int((pos.x - grid.origin.x) / grid.cellSize);
    cell.y = int((pos.y - grid.origin.y) / grid.cellSize);
    cell.z = int((pos.z - grid.origin.z) / grid.cellSize);
    cell.x = dmax(0, dmin(cell.x, grid.gridSize.x - 1));
    cell.y = dmax(0, dmin(cell.y, grid.gridSize.y - 1));
    cell.z = dmax(0, dmin(cell.z, grid.gridSize.z - 1));
    return cell;
}

__global__ void kernelAssignParticlesToGrid(
    float4* positions, int* particleCellHashes, int* particleIndices, int numParticles, Grid grid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    float3 pos = make_float3(positions[i].x, positions[i].y, positions[i].z);
    int hash = getCellHash(pos, grid);
    particleCellHashes[i] = hash;
    particleIndices[i] = i;
}

__global__ void extractPositionsKernel(const Particle* particles, float4* positions, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float3 p = particles[i].position;
    positions[i] = make_float4(p.x, p.y, p.z, 1.0f);
}

void initializeGrid(Grid& grid, const FluidParams& params, int numParticles) {
    grid.cellSize = params.smoothingRadius * GRID_CELL_SIZE_FACTOR;
    grid.origin = make_float3(-2.0f, -2.0f, -2.0f);
    grid.gridSize.x = int((4.0f) / grid.cellSize) + 1;
    grid.gridSize.y = int((4.0f) / grid.cellSize) + 1;
    grid.gridSize.z = int((4.0f) / grid.cellSize) + 1;
    int numCells = grid.gridSize.x * grid.gridSize.y * grid.gridSize.z;
    cudaMalloc(&grid.cellStart, numCells * sizeof(int));
    cudaMalloc(&grid.cellEnd, numCells * sizeof(int));
    cudaMalloc(&grid.particleIndices, numParticles * sizeof(int));
}

void freeGrid(Grid& grid) {
    cudaFree(grid.cellStart);
    cudaFree(grid.cellEnd);
    cudaFree(grid.particleIndices);
}

void assignParticlesToGrid(
    float4* positions, int* particleCellHashes, int* particleIndices, int numParticles, const Grid& grid, cudaStream_t stream)
{
    int block = 128;
    int gridDim = (numParticles + block - 1) / block;
    kernelAssignParticlesToGrid<<<gridDim, block, 0, stream>>>(positions, particleCellHashes, particleIndices, numParticles, grid);
}

__global__ void kernelBuildCellRanges(
    int* particleCellHashes, int* particleIndices, int numParticles, int* cellStart, int* cellEnd, int numCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    int hash = particleCellHashes[i];
    if (i == 0) cellStart[hash] = 0;
    else {
        int prevHash = particleCellHashes[i - 1];
        if (hash != prevHash) {
            cellEnd[prevHash] = i;
            cellStart[hash] = i;
        }
    }
    if (i == numParticles - 1) cellEnd[hash] = numParticles;
}

void buildCellRanges(
    int* particleCellHashes, int* particleIndices, int numParticles, Grid& grid, cudaStream_t stream)
{
    int numCells = grid.gridSize.x * grid.gridSize.y * grid.gridSize.z;
    cudaMemsetAsync(grid.cellStart, 0xff, numCells * sizeof(int), stream);
    cudaMemsetAsync(grid.cellEnd, 0xff, numCells * sizeof(int), stream);

    // Sort by cell hash
    thrust::device_ptr<int> hashPtr(particleCellHashes);
    thrust::device_ptr<int> idxPtr(particleIndices);
    thrust::sort_by_key(thrust::cuda::par.on(stream), hashPtr, hashPtr + numParticles, idxPtr);

    int block = 128;
    int gridDim = (numParticles + block - 1) / block;
    kernelBuildCellRanges<<<gridDim, block, 0, stream>>>(particleCellHashes, particleIndices, numParticles, grid.cellStart, grid.cellEnd, numCells);
}

__global__ void integrateGrid(
    float4* positionsOut, Particle* particles, int count, FluidParams params,
    const Grid grid, const int* cellStart, const int* cellEnd, const int* particleIndices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle& p = particles[i];
    float3 pos = p.position;
    int3 cell = getCellCoord(pos, grid);

    // Density and Pressure
    float density = 0.0f;
    struct DensityCtx {
        Particle* particles;
        float3 pos;
        float h;
        float* density;
        const int* cellStart;
        const int* cellEnd;
        const int* particleIndices;
        int selfIdx;
        float particleMass;
    } densityCtx = {particles, pos, params.smoothingRadius, &density, cellStart, cellEnd, particleIndices, i, params.particleMass};

    auto densityNeighbor = [] __device__ (int neighborHash, void* userData) {
        auto* c = (DensityCtx*)userData;
        int start = c->cellStart[neighborHash];
        int end = c->cellEnd[neighborHash];
        if (start == -1 || end == -1) return;
        for (int idx = start; idx < end; ++idx) {
            int j = c->particleIndices[idx];
            float3 rij = c->pos - c->particles[j].position;
            float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
            if (r2 < c->h * c->h)
                *(c->density) += c->particleMass * poly6(r2, c->h);
        }
    };
    forEachNeighborCell(cell, grid, densityNeighbor, &densityCtx);

    p.density = density;
    p.pressure = params.pressureStiffness * (density - params.restDensity);

    // Forces
    float3 force = params.gravity * p.density;
    struct ForceCtx {
        Particle* particles;
        float3 pos;
        float3 vel;
        float pressure;
        float density;
        float h;
        float3* force;
        const int* cellStart;
        const int* cellEnd;
        const int* particleIndices;
        int selfIdx;
        float particleMass;
        float viscosity;
    } forceCtx = {particles, pos, p.velocity, p.pressure, p.density, params.smoothingRadius, &force, cellStart, cellEnd, particleIndices, i, params.particleMass, params.viscosity};

    auto forceNeighbor = [] __device__ (int neighborHash, void* userData) {
        auto* c = (ForceCtx*)userData;
        int start = c->cellStart[neighborHash];
        int end = c->cellEnd[neighborHash];
        if (start == -1 || end == -1) return;
        for (int idx = start; idx < end; ++idx) {
            int j = c->particleIndices[idx];
            if (j == c->selfIdx) continue;
            Particle& pj = c->particles[j];
            float3 rij = c->pos - pj.position;
            float r = length(rij);
            if (r < c->h && r > 1e-5f) {
                // Pressure
                float3 grad = spikyGrad(rij, c->h);
                float pressureTerm = (c->pressure + pj.pressure) / (2.0f * pj.density);
                *(c->force) += -c->particleMass * pressureTerm * grad;

                // Viscosity
                float3 velDiff = pj.velocity - c->vel;
                float viscKernel = (45.0f / (M_PI * pow(c->h, 6))) * (c->h - r);
                float3 visc = c->viscosity * c->particleMass * velDiff / pj.density;
                *(c->force) += visc * viscKernel;
            }
            // Collision
            float particleRadius = c->h * 0.5f;
            float minDist = 2.0f * particleRadius;
            float restitution = 0.2f;
            if (r < minDist && r > 1e-5f) {
                float3 n = rij / r;
                float3 relVelVec = c->vel - pj.velocity;
                float relVel = relVelVec.x * n.x + relVelVec.y * n.y + relVelVec.z * n.z;
                float penetration = minDist - r;
                float forceMag = 0.025f * penetration - (restitution * relVel);
                if (forceMag < 0) forceMag = 0;
                *(c->force) += n * forceMag;
            }
        }
    };
    forEachNeighborCell(cell, grid, forceNeighbor, &forceCtx);

    // Integrate
    p.velocity += (force / p.density) * params.timeStep;
    p.position += p.velocity * params.timeStep;

    // Collision with bounds
    float3 min = params.boundsMin;
    float3 max = params.boundsMax;
    for (int j = 0; j < 3; ++j) {
        if (((&p.position.x)[j] < (&min.x)[j]) || ((&p.position.x)[j] > (&max.x)[j])) {
            (&p.velocity.x)[j] *= -0.5f;
            (&p.position.x)[j] = fmaxf(fminf((&p.position.x)[j], (&max.x)[j]), (&min.x)[j]);
        }
    }

    // Write to VBO
    positionsOut[i] = make_float4(p.position.x, p.position.y, p.position.z, 1.0f);
}

void simulateParticles(
    float4* posVBO, int count, FluidParams params,
    bool resetSimRequested, bool paused)
{
    static Particle* d_particles = nullptr;
    static bool initialized = false;
    static Grid grid;
    static int* d_cellHashes = nullptr;
    static int* d_particleIndices = nullptr;
    static float4* d_positions = nullptr;

    // Free Memory when updating Particle Count
    if (resetSimRequested) {
        if (d_particles) cudaFree(d_particles);
        if (d_positions) cudaFree(d_positions);
        if (d_cellHashes) cudaFree(d_cellHashes);
        if (d_particleIndices) cudaFree(d_particleIndices);
        freeGrid(grid);
        initialized = false;
    }

    if (!initialized) {
        cudaMalloc(&d_particles, count * sizeof(Particle));
        cudaMalloc(&d_positions, count * sizeof(float4));
        Particle* temp = new Particle[count];
        float4* tempPos = new float4[count];

        float spacing = 0.05f;
        int layers = static_cast<int>(ceil(cbrt((double)count)));

        for (int n = 0; n < count; ++n) {
            int layer = n / (layers * layers);
            int indexInLayer = n % (layers * layers);
            int i = indexInLayer % layers;
            int k = indexInLayer / layers;

            float fx = params.boundsMin.x + spacing * (i + 0.5f);
            float fy = params.boundsMin.y + spacing * (layer + 0.5f);
            float fz = params.boundsMin.z + spacing * (k + 0.5f);

            float offset = 0.01f * layer;
            fx += offset;
            fz += offset;

            temp[n].position = make_float3(fx, fy, fz);
            temp[n].velocity = make_float3(0, 0, 0);
            temp[n].density = params.restDensity;
            temp[n].pressure = 0.0f;
            tempPos[n] = make_float4(fx, fy, fz, 1.0f);
        }

        std::cout << "Initialized particles: " << count << " (should be " << count << ")" << std::endl;

        cudaMemcpy(d_particles, temp, count * sizeof(Particle), cudaMemcpyHostToDevice);
        cudaMemcpy(d_positions, tempPos, count * sizeof(float4), cudaMemcpyHostToDevice);
        delete[] temp;
        delete[] tempPos;

        // Initial copy of positions into VBO
        cudaMemcpy(posVBO, d_positions, count * sizeof(float4), cudaMemcpyDeviceToDevice);

        initializeGrid(grid, params, count);
        cudaMalloc(&d_cellHashes, count * sizeof(int));
        cudaMalloc(&d_particleIndices, count * sizeof(int));
        initialized = true;
    }

    if (!paused) {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;

        // Update positions
        extractPositionsKernel<<<blocks, threads>>>(d_particles, d_positions, count);
        assignParticlesToGrid(d_positions, d_cellHashes, d_particleIndices, count, grid, 0);
        buildCellRanges(d_cellHashes, d_particleIndices, count, grid, 0);
        integrateGrid<<<blocks, threads>>>(posVBO, d_particles, count, params, grid, grid.cellStart, grid.cellEnd, d_particleIndices);
        extractPositionsKernel<<<blocks, threads>>>(d_particles, posVBO, count);
    }
}
