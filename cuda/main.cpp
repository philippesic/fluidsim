#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include "simulation.cuh"
#include "fluid.h"

const int NUM_PARTICLES = 1024;
GLuint vbo, vao, shaderProgram;
struct cudaGraphicsResource* cudaVBO = nullptr;

const char* vertexShaderSrc = R"glsl(
#version 330 core
layout(location = 0) in vec4 inPos;
void main() {
    gl_Position = inPos;
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

void createVBO(GLuint* vbo, struct cudaGraphicsResource** cudaResource) {
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaError_t err = cudaGraphicsGLRegisterBuffer(cudaResource, *vbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsGLRegisterBuffer error: " << cudaGetErrorString(err) << std::endl;
    }
}

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA + OpenGL Interop", nullptr, nullptr);
    if (!window) return -1;
    glfwMakeContextCurrent(window);
    glewExperimental = true;
    if (glewInit() != GLEW_OK) return -1;

    // Force CUDA context creation after OpenGL context is current
    cudaFree(0);

    shaderProgram = createProgram(vertexShaderSrc, fragmentShaderSrc);
    if (!shaderProgram) return -1;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    createVBO(&vbo, &cudaVBO);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    while (!glfwWindowShouldClose(window)) {
        // CUDA: Map VBO, update with kernel, unmap
        float4* dptr = nullptr;
        cudaGraphicsMapResources(1, &cudaVBO, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cudaVBO);

        // Call your CUDA kernel to update positions
        simulateParticles(dptr, NUM_PARTICLES, Fluid::water);

        cudaGraphicsUnmapResources(1, &cudaVBO, 0);

        // OpenGL: Render
        glViewport(0, 0, 800, 600);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
        glBindVertexArray(0);
        glUseProgram(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cudaVBO);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
