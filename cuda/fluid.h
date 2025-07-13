#pragma once
#include <cuda_runtime.h>

struct FluidParams {
    float restDensity;
    float viscosity;
    float pressureStiffness;
    float3 gravity;
    float smoothingRadius;
    float particleMass;
    float timeStep;
    float3 boundsMin;
    float3 boundsMax;
};

//Fluids
namespace Fluid {

    static FluidParams water = {
        1000.0f,
        1.0f,
        1.0f,
        make_float3(0.0f, -9.8f, 0.0f),
        0.1f,
        1.0f,
        0.0025f,
        make_float3(-0.5f,-0.5f,-0.5f),
        make_float3(0.5f,0.5f,0.5f)
    };

    static FluidParams lava = {};
    static FluidParams honey = {};
}

