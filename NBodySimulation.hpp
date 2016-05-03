#ifndef NBodySimulation_H
#define NBodySimulation_H

#ifdef __APPLE__
# define __gl_h_
# define GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Registry.hpp"
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "computations.h"

class NBodySimulation
{
public:
    NBodySimulation();
    ~NBodySimulation();

    void init();
    void render();

private:
    void initNBodyPositions();

private: // OpenGL vertex buffers
    GLuint vao;
    GLuint posBodiesBuffer;
    GLuint glShaderV, glShaderF;
    GLuint glProgram;

private:
    struct cudaGraphicsResource *cuda_pos_resource;

private:
    glm::vec4* N_Bodies;
    float4* dev_bodies;
    float3* dev_velocities;
    float3* dev_acceleration;
};

#endif
