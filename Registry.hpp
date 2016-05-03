#ifndef Registry_h
#define Registry_h

#define __DEBUG__
#define GLFW_INCLUDE_GLCOREARB

#include <glm/glm.hpp>
#include <fstream>

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

#define checkCudaErrors(call) {										             \
    cudaError err = call;												         \
    if(err != cudaSuccess) {											         \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	         \
            __FILE__, __LINE__, cudaGetErrorString(err));				         \
        exit(1);														         \
    }																	         \
} while (0)

#endif
