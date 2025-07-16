#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <glm.hpp>
#include <gtc/matrix_transform.inl>
#include <gtc/type_ptr.hpp>
#include "simulation.cuh"
#include "fluid.h"
#include "ui.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

int num_particles = 16384;
GLuint vbo, vao, shaderProgram;
struct cudaGraphicsResource* cudaVBO = nullptr;

double lastX = 400, lastY = 300;
float yaw = 0.0f, pitch = 0.0f;
bool firstMouse = true;
float radius = 2.0f;
bool cameraControl = false;

static bool resetSimRequested = false;
const char* vertexShaderSrc = R"glsl(
#version 330 core
layout(location = 0) in vec4 inPos;
uniform mat4 uProjection;
uniform mat4 uView;
void main() {
    gl_Position = uProjection * uView * inPos;
    gl_PointSize = 10.0;
}
)glsl";

const char* fragmentShaderSrc = R"glsl(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(0.2, 0.6, 1.0, 1.0);
}
)glsl";

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetShaderInfoLog(shader, 512, nullptr, buffer);
        std::cerr << "Shader compile error: " << buffer << "\n";
        return 0;
    }
    return shader;
}

GLuint createProgram(const char* vertSrc, const char* fragSrc) {
    GLuint program = glCreateProgram();
    GLuint vert = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    if (!vert || !frag) return 0;
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetProgramInfoLog(program, 512, nullptr, buffer);
        std::cerr << "Program link error: " << buffer << "\n";
        return 0;
    }
    glDeleteShader(vert);
    glDeleteShader(frag);
    return program;
}

void createVBO(GLuint* vbo, struct cudaGraphicsResource** cudaResource, int numParticles) {
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaError_t err = cudaGraphicsGLRegisterBuffer(cudaResource, *vbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsGLRegisterBuffer error: " << cudaGetErrorString(err) << std::endl;
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!cameraControl) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = true;
        return;
    }
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    float xoffset = float(xpos - lastX);
    float yoffset = float(lastY - ypos);
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.2f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // Clamp radius
    extern float radius;
    radius -= static_cast<float>(yoffset) * 0.2f;
    if (radius < 0.5f) radius = 0.5f;
    if (radius > 10.0f) radius = 10.0f;
}

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(800, 600, "FluidSim", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glewExperimental = true;
    if (glewInit() != GLEW_OK) return -1;

    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    cudaFree(0);

    shaderProgram = createProgram(vertexShaderSrc, fragmentShaderSrc);
    if (!shaderProgram) return -1;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    createVBO(&vbo, &cudaVBO, num_particles);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Initial projection
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 10.0f);

    // ImGui setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    FluidUI fluidUI;

    while (!glfwWindowShouldClose(window)) {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        projection = glm::perspective(glm::radians(45.0f), float(width) / float(height), 0.1f, 10.0f);

        // Handle rmb Camera
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!cameraControl) {
                cameraControl = true;
                firstMouse = true;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
        } else {
            if (cameraControl) {
                cameraControl = false;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
        }

        // Camera orbit
        float camX = radius * sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        float camY = radius * sin(glm::radians(pitch));
        float camZ = radius * cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        glm::vec3 simulationCenter(0.0f, 0.0f, 0.0f);
        glm::vec3 cameraPos = simulationCenter + glm::vec3(camX, camY, camZ);
        glm::mat4 view = glm::lookAt(cameraPos, simulationCenter, glm::vec3(0.0f, 1.0f, 0.0f));

        float4* dptr = nullptr;
        cudaGraphicsMapResources(1, &cudaVBO, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cudaVBO);

        // Update simulation
        simulateParticles(dptr, num_particles, fluidUI.params, fluidUI.shouldReset(), fluidUI.isPaused());
        fluidUI.clearReset();

        cudaGraphicsUnmapResources(1, &cudaVBO, 0);

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        GLint projLoc = glGetUniformLocation(shaderProgram, "uProjection");
        GLint viewLoc = glGetUniformLocation(shaderProgram, "uView");
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, num_particles);
        glBindVertexArray(0);
        glUseProgram(0);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        ImGui::Begin("Simulation Settings");
        if (ImGui::SliderInt("Particle Count", &num_particles, 1, 10000)) {
            fluidUI.resetRequested = true;
            cudaGraphicsUnregisterResource(cudaVBO);
            glDeleteBuffers(1, &vbo);
            createVBO(&vbo, &cudaVBO, num_particles);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
        ImGui::End();

        fluidUI.render();
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cudaVBO);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shaderProgram);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
