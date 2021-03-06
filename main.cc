#ifdef __APPLE__
# define __gl_h_
# define GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
#endif

#include <stdlib.h>
#include <stdio.h>
#include "gpu_simulation.hpp"
#include "cpu_simulation.hpp"
#include "gpu_bh_simulation.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

void error_callback(int error, const char* description);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double x, double y);

bool cursor = false;
double cursorX;
double cursorY;


int main(void)
{
    GLFWwindow* window;

    if(!glfwInit()) {
        printf("Error: glfwInit\n");
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1024, 1024, "N-body simulation", NULL, NULL);
    if(!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetKeyCallback(window, key_callback);
    //glfwSetMouseButtonCallback(window, mouse_button_callback);
    //glfwSetCursorPosCallback(window, cursor_position_callback);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

#ifdef __DEBUG__
    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
#endif

    //AbstractSimulation* simulation = new GPU_Simulation;
    //AbstractSimulation* simulation = new CPU_Simulation;
    AbstractSimulation* simulation = new GPU_BarnesHut_Simulation;

    simulation->setPotentialFieldRendering(false);
    simulation->init(_2D_SIMULATION_);

    #ifndef _DEBUG_THRUST_
    while (!glfwWindowShouldClose(window))
    {
    #endif
        glfwGetFramebufferSize(window, &AbstractSimulation::width, &AbstractSimulation::height);

        simulation->render();

        glfwSwapBuffers(window);
        glfwPollEvents();
    #ifndef _DEBUG_THRUST_
    }
    #else
    getchar();
    #endif

    return EXIT_SUCCESS;
}

void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;

    float eps = 50.0f;

    glm::vec3 prevCameraPos = AbstractSimulation::cameraPos;

    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GL_TRUE); break;
        case GLFW_KEY_W: // move up along the sphere
            AbstractSimulation::cameraPos.y = fmod(AbstractSimulation::cameraPos.y + eps, 2 * M_PI);
            break;
        case GLFW_KEY_S: // move down along the sphere
            AbstractSimulation::cameraPos.y = fmod(AbstractSimulation::cameraPos.y - eps, 2 * M_PI);
            break;
        case GLFW_KEY_A: // move left along the sphere
            AbstractSimulation::cameraPos.z = fmod(AbstractSimulation::cameraPos.z + eps, 2 * M_PI);
            break;
        case GLFW_KEY_D: // move right along the sphere
            AbstractSimulation::cameraPos.z = fmod(AbstractSimulation::cameraPos.z - eps, 2 * M_PI);
            break;
        case GLFW_KEY_Q: AbstractSimulation::cameraPos.x -= eps; break; // decrease Radius
        case GLFW_KEY_E: AbstractSimulation::cameraPos.x += eps; break; // increase Radius
        default:
            break;
    }
}
