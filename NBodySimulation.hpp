#ifndef NBodySimulation_H
#define NBodySimulation_H

#include <cuda.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Registry.hpp"
#include <GLFW/glfw3.h>
#include <OpenGL/gl3.h>

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

    glm::vec3* N_Bodies;
};

#endif
