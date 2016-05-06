#ifndef AbstractSimulation_H
#define AbstractSimulation_H

#ifdef __APPLE__
# define __gl_h_
# define GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
#endif

#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>

#define __DEBUG__
#define GLFW_INCLUDE_GLCOREARB

#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#define checkCudaErrors(call) {										             \
    cudaError err = call;												         \
    if(err != cudaSuccess) {											         \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	         \
            __FILE__, __LINE__, cudaGetErrorString(err));				         \
        exit(1);														         \
    }																	         \
}

char* loadFile(const char *filename);
glm::vec3 sphericalToCartesian(glm::vec3 vec);

#define _2D_SIMULATION_ 2
#define _3D_SIMULATION_ 3

class AbstractSimulation
{
public:
    AbstractSimulation();
    ~AbstractSimulation();

    virtual void init(int dimensions);
    virtual void render() = 0;

protected: // OpenGL vertex buffers
    GLuint vao;
    GLuint posBodiesBuffer;
    GLuint glShaderV, glShaderF;
    GLuint glProgram;

protected:
    void initNBodyPositions(int dimensions);

protected:
    glm::vec4* bodies; // w -- mass
    glm::vec3* velocities;

public:
    static unsigned int N;
    static glm::vec2 pMouse;

    static glm::vec3 cameraPos; // in spherical coordinates (r, theta, phi)
    static glm::vec3 centerOfMass;
    static glm::vec3 upVector;

    static int width;
    static int height;
};

#endif
