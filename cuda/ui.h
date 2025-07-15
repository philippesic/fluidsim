#pragma once
#include "fluid.h"

struct FluidUI {
    FluidParams params;
    bool resetRequested = false;
    bool paused = false;

    FluidUI() : params(Fluid::water) {}

    void render();
    bool shouldReset() const { return resetRequested; }
    void clearReset() { resetRequested = false; }
    bool isPaused() const { return paused; }
    void setPaused(bool p) { paused = p; }
};