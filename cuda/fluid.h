#pragma once
#include <cuda_runtime.h>

struct FluidParams {
    float restDensity;
    float viscosity;
    float pressureStiffness;
    float3 gravity;
};

//Fluids
namespace Fluid {

    static FluidParams water = {
        1000.0f,
        0.1f,
        3.0f,
        make_float3(0.0f, -9.8f, 0.0f)
    };

    static FluidParams lava = {};
    static FluidParams honey = {};
}

