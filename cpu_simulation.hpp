#ifndef CPU_SIMULATION_H
#define CPU_SIMULATION_H

#include "abstractsimulation.hpp"

class CPU_Simulation : public AbstractSimulation
{
public:
    CPU_Simulation();
    ~CPU_Simulation();

    virtual void init(int dimensions);
    virtual void render();

protected:
    void integrateSystem(glm::vec4* bodies, glm::vec3* velocities, glm::vec3* acceleration, unsigned int N);
    void computeCenterOfMass(glm::vec4* bodies, unsigned int N);

protected:
    glm::vec3* acceleration;
};

#endif
