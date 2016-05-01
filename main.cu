#ifdef __APPLE__
# define __gl_h_
# define GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
#endif

#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "NBodySimulation.hpp"
#include "Registry.hpp"

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

    NBodySimulation simulation;
    simulation.init();

    while (!glfwWindowShouldClose(window))
    {
        glfwGetFramebufferSize(window, &Registry::width, &Registry::height);

        simulation.render();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaDeviceReset();
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

    float eps = 5e-2f;

    glm::vec3 prevCameraPos = Registry::cameraPos;

    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GL_TRUE); break;
        case GLFW_KEY_W: // move up along the sphere
            Registry::cameraPos.y = fmod(Registry::cameraPos.y - eps, 2 * M_PI);
            break;
        case GLFW_KEY_S: // move down along the sphere
            Registry::cameraPos.y = fmod(Registry::cameraPos.y + eps, 2 * M_PI);
            break;
        case GLFW_KEY_A: // move left along the sphere
            Registry::cameraPos.z = fmod(Registry::cameraPos.z - eps, 2 * M_PI);
            break;
        case GLFW_KEY_D: // move right along the sphere
            Registry::cameraPos.z = fmod(Registry::cameraPos.z + eps, 2 * M_PI);
            break;
        case GLFW_KEY_Q: Registry::cameraPos.x -= eps; break; // decrease Radius
        case GLFW_KEY_E: Registry::cameraPos.x += eps; break; // increase Radius
        #ifdef __DEBUG__
            case GLFW_KEY_T: changeCenterOfMass(glm::vec3(0.0f,0.0f,0.0f));
        #endif
        default:
            break;
    }


#ifdef __DEBUG__
    printf("cameraPos: %f %f %f\n", Registry::cameraPos.x, Registry::cameraPos.y, Registry::cameraPos.z);
#endif
}
