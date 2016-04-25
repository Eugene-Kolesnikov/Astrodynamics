#ifndef NBodySimulation_H
#define NBodySimulation_H

#include <cuda.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Registry.hpp"

class NBodySimulation
{
public:
    NBodySimulation();
    ~NBodySimulation();

    void init();
    void render();

private:
    void initNBodyPositions();

private:
};

#endif
