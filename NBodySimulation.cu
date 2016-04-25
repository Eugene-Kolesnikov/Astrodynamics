#include "NBodySimulation.hpp"
#include <glm/gtc/random.hpp>
#include <GLFW/glfw3.h>

NBodySimulation::NBodySimulation()
{

}

NBodySimulation::~NBodySimulation()
{

}

void NBodySimulation::init()
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void NBodySimulation::render()
{
    /*glm::mat4 Projection = glm::perspective(45.5f, (float)Registry::width / Registry::height, 0.1f, 3000.0f);
    glm::mat4 RotationPitch = glm::rotate(glm::mat4(1.0f), -Registry::pitch, glm::vec3(1,0,0));
    glm::mat4 RotationYaw = glm::rotate(glm::mat4(1.0f), -Registry::yaw, glm::vec3(0,1,0));
    glm::mat4 Translate = glm::translate(glm::mat4(1.0f),Registry::cameraPos);
    glm::mat4 PV = Projection * RotationPitch * RotationYaw * Translate;*/
    float ratio = Registry::width / (float) Registry::height;
    glViewport(0, 0, Registry::width, Registry::height);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef((float) glfwGetTime() * 50.f, 0.f, 0.f, 1.f);
    glBegin(GL_TRIANGLES);
    glColor3f(1.f, 0.f, 0.f);
    glVertex3f(-0.6f, -0.4f, 0.f);
    glColor3f(0.f, 1.f, 0.f);
    glVertex3f(0.6f, -0.4f, 0.f);
    glColor3f(0.f, 0.f, 1.f);
    glVertex3f(0.f, 0.6f, 0.f);
    glEnd();
}

void NBodySimulation::initNBodyPositions()
{
    /* glm::vec4(glm::ballRand(1.0f), 1);
        Generate a random 3D vector which coordinates are regulary distributed within the volume of a ball of a given radius.
        (http://glm.g-truc.net/0.9.7/glm-0.9.7.pdf) */
}
