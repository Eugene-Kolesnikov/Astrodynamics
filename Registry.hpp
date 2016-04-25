#ifndef Registry_h
#define Registry_h

#define __DEBUG__
#define GLFW_INCLUDE_GLCOREARB

#include <glm/glm.hpp>

class Registry
{
public:
    static glm::vec2 pMouse;
    static glm::vec3 cameraPos;
    static float pitch;
    static float yaw;
    static int width;
    static int height;
};

char* loadFile(const char *filename);

#endif
