#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "simulation.cuh"
#include "fluid.h"
#include "utils.h"
#define _USE_MATH_DEFINES
#include <math.h>



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

__global__ void integrate(float4* positionsOut, Particle* particles, int count, FluidParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle& p = particles[i];

    // Density and Pressure
    float density = 0.0f;
    float pressure = 0.0f;
    for (int j = 0; j < count; ++j) {
        float3 rij = p.position - particles[j].position;
        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
        if (r2 < params.smoothingRadius * params.smoothingRadius) {
            density += params.particleMass * poly6(r2, params.smoothingRadius);
        }
    }
    p.density = density;
    p.pressure = params.pressureStiffness * (density - params.restDensity);

    // Forces
    float3 force = params.gravity * p.density;
    for (int j = 0; j < count; ++j) {
        if (i == j) continue;
        Particle& pj = particles[j];
        float3 rij = p.position - pj.position;
        float r = length(rij);
        if (r < params.smoothingRadius && r > 1e-5f) {
            // Pressure force
            float3 grad = spikyGrad(rij, params.smoothingRadius);
            float pressureTerm = (p.pressure + pj.pressure) / (2.0f * pj.density);
            force += -params.particleMass * pressureTerm * grad;

            // Viscosity force
            float3 velDiff = pj.velocity - p.velocity;
            float viscKernel = (45.0f / (M_PI * pow(params.smoothingRadius, 6))) * (params.smoothingRadius - r);
            float3 visc = params.viscosity * params.particleMass * velDiff / pj.density;
            force += visc * viscKernel;
        }
    }

    // Particle Collision
    float particleRadius = params.smoothingRadius * 0.5f;
    float minDist = 2.0f * particleRadius;
	// Lower to soften bounce
    float restitution = 0.2f;
    for (int j = 0; j < count; ++j) {
        if (i == j) continue;
        Particle& pj = particles[j];
        float3 rij = p.position - pj.position;
        float dist = length(rij);
        if (dist < minDist && dist > 1e-5f) {
            float3 n = rij / dist;
            float3 relVelVec = p.velocity - pj.velocity;
            float relVel = relVelVec.x * n.x + relVelVec.y * n.y + relVelVec.z * n.z;
            float penetration = minDist - dist;
            // Lower to soften repulsion
            float forceMag = 0.025f * penetration - (restitution * relVel);
            if (forceMag < 0) forceMag = 0;
            force += n * forceMag;
        }
    }

    // Integrate
    p.velocity += (force / p.density) * params.timeStep;
    p.position += p.velocity * params.timeStep;

    // Collision with bounds
    float3 min = params.boundsMin;
    float3 max = params.boundsMax;
    for (int j = 0; j < 3; ++j) {
        if (((&p.position.x)[j] < (&min.x)[j]) || ((&p.position.x)[j] > (&max.x)[j])) {
            (&p.velocity.x)[j] *= -0.5f; // bounce
            (&p.position.x)[j] = fmaxf(fminf((&p.position.x)[j], (&max.x)[j]), (&min.x)[j]);
        }
    }

    // Write to VBO
    positionsOut[i] = make_float4(p.position.x, p.position.y, p.position.z, 1.0f);
}

void simulateParticles(float4* posVBO, int count, FluidParams params) {
    static Particle* d_particles = nullptr;
    static bool initialized = false;

    if (!initialized) {
        cudaMalloc(&d_particles, count * sizeof(Particle));
        Particle* temp = new Particle[count];

        // Scatter particles
        float spacing = params.smoothingRadius * 0.55f;
        int gridX = int((params.boundsMax.x - params.boundsMin.x) / spacing);
        int gridY = int((params.boundsMax.y - params.boundsMin.y) / spacing);
        int gridZ = int((params.boundsMax.z - params.boundsMin.z) / spacing);

        int n = 0;
        for (int z = 0; z < gridZ && n < count; ++z) {
            for (int y = 0; y < gridY && n < count; ++y) {
                for (int x = 0; x < gridX && n < count; ++x) {
                    float px = params.boundsMin.x + spacing * (x + 0.5f);
                    float py = params.boundsMin.y + spacing * (y + 0.5f);
                    float pz = params.boundsMin.z + spacing * (z + 0.5f);
                    temp[n].position = make_float3(px, py, pz);
                    temp[n].velocity = make_float3(0,0,0);
                    temp[n].density = params.restDensity;
                    temp[n].pressure = 0.0f;
                    ++n;
                }
            }
        }

        cudaMemcpy(d_particles, temp, count * sizeof(Particle), cudaMemcpyHostToDevice);
        delete[] temp;
        initialized = true;
    }

    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    integrate <<<blocks, threads>>> (posVBO, d_particles, count, params);
}
