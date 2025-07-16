#include "ui.h"
#include "imgui.h"

void FluidUI::render() {
    ImGui::Begin("Fluid Controls");
    ImGui::SliderFloat("Rest Density", &params.restDensity, 0.0f, 2000.0f);
    ImGui::SliderFloat("Viscosity", &params.viscosity, 0.0f, 10.0f);
    ImGui::SliderFloat("Pressure Stiffness", &params.pressureStiffness, 0.1f, 10.0f);
    ImGui::SliderFloat3("Gravity", (float*)&params.gravity, -20.0f, 20.0f);
    ImGui::SliderFloat("Smoothing Radius", &params.smoothingRadius, 0.01f, 2.0f);
    ImGui::SliderFloat("Particle Mass", &params.particleMass, 0.01f, 10.0f);
    ImGui::SliderFloat("Time Step", &params.timeStep, 0.0001f, 0.02f, "%.5f");
    ImGui::SliderFloat3("Bounds Min", (float*)&params.boundsMin, -2.0f, 0.0f);
    ImGui::SliderFloat3("Bounds Max", (float*)&params.boundsMax, 0.0f, 2.0f);

    if (ImGui::Button(paused ? "Unpause Simulation" : "Pause Simulation")) {
        paused = !paused;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset Simulation")) {
        resetRequested = true;
        paused = false;
    }
    ImGui::End();
}