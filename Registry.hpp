#ifndef Registry_h
#define Registry_h

#define __DEBUG__
#define GLFW_INCLUDE_GLCOREARB

#include <glm/glm.hpp>

class Registry
{
public:
    static glm::vec2 pMouse;

    static glm::vec3 cameraPos; // in spherical coordinates (r, theta, phi)
    static glm::vec3 centerOfMass;
    static glm::vec3 upVector;

    static int width;
    static int height;
};

char* loadFile(const char *filename);

glm::vec3 sphericalToCartesian(glm::vec3 vec);
glm::vec3 cartesianToSpherical(glm::vec3 vec);
void changeCenterOfMass(glm::vec3 center);

#endif
