#ifndef CPU_COMPUTATIONS_H
#define CPU_COMPUTATIONS_H

#include "abstractsimulation.hpp"

void cpu_initVelocities(glm::vec4* bodies, glm::vec3* velocities, glm::vec3* acceleration);
void cpu_integrateSystem(glm::vec4* bodies, glm::vec3* velocities, glm::vec3* acceleration);
void cpu_computeCenterOfMass(glm::vec4* bodies);

#endif
