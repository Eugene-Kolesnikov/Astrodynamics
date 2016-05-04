#include "Registry.hpp"

glm::vec2 Registry::pMouse = glm::vec2(0.0f, 0.0f);

glm::vec3 Registry::cameraPos = glm::vec3(1200.0f, M_PI_2, 0.0f); // in spherical coordinates (r, theta, phi)
glm::vec3 Registry::centerOfMass = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 Registry::upVector = glm::vec3(0.0f, 0.0f, 1.0f);

int Registry::width = 0;
int Registry::height = 0;

#include <fstream>
#include <sstream>

char* loadFile(const char *filename) {
    char* data;
    int len;
    std::ifstream ifs(filename, std::ifstream::in);
    if(ifs.is_open() == false) {
        printf("File not open!\n");
    }
    ifs.seekg(0, std::ios::end);
    len = (int)ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data = new char[len + 1];
    ifs.read(data, len);
    data[len] = 0;
    ifs.close();
    return data;
}

glm::vec3 sphericalToCartesian(glm::vec3 vec)
{
    return glm::vec3(
        vec.x * sin(vec.y) * cos(vec.z),
        vec.x * sin(vec.y) * sin(vec.z),
        vec.x * cos(vec.y)
    );
}
