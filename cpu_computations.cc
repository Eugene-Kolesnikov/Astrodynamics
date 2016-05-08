#include "cpu_computations.hpp"
#include <glm/gtc/type_ptr.hpp>

#define _GAMMA_ 6.67408
#define _TAU_ 5.0e-3f
#define _EPS2_ 70

glm::vec3 bodyBodyInteraction(glm::vec4 bi, glm::vec4 bj, glm::vec3 ai) {
    glm::vec3 r;
    // r_ij [3 FLOPS]
    r.x = bj.x - bi.x; r.y = bj.y - bi.y; r.z = bj.z - bi.z;
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    float distSqr = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
    float distSqr_eps = r.x * r.x + r.y * r.y + r.z * r.z + _EPS2_;
    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth_eps = distSqr_eps * distSqr_eps * distSqr_eps;
    float invDistCube = 1.0f/sqrtf(distSixth_eps);
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * _GAMMA_ * invDistCube;
    //a_i= a_i+s*r_ij[6FLOPS]
    ai.x += r.x * s; ai.y += r.y * s; ai.z += r.z * s;
    return ai;
}

void calculateForces(glm::vec4* bodies, glm::vec3* acceleration)
{
    for(int i = 0; i < AbstractSimulation::N; ++i) {
        glm::vec3 acc(0.0f,0.0f,0.0f);
        for(int j = 0; j < AbstractSimulation::N; ++j) {
            acc = bodyBodyInteraction(bodies[i], bodies[j], acc);
        }
        acceleration[i] = acc;
    }
}

void cpu_initVelocities(glm::vec4* bodies, glm::vec3* velocities, glm::vec3* acceleration)
{
    calculateForces(bodies, acceleration);
    for(int i = 0; i < AbstractSimulation::N; ++i)
        velocities[i] += (_TAU_ / 2.0f) * acceleration[i];
}

void updatePositions(glm::vec4* bodies, glm::vec3* velocities)
{
    for(int i = 0; i < AbstractSimulation::N; ++i)
    {
        glm::vec4 vel(velocities[i].x, velocities[i].y, velocities[i].z, 0.0f);
        bodies[i] += _TAU_ * vel;
    }
}

void updateVelocities(glm::vec3* velocities, glm::vec3* acceleration)
{
    for(int i = 0; i < AbstractSimulation::N; ++i)
        velocities[i] += _TAU_ * acceleration[i];
}

void cpu_integrateSystem(glm::vec4* bodies, glm::vec3* velocities, glm::vec3* acceleration)
{
    updatePositions(bodies, velocities);
    calculateForces(bodies, acceleration);
    updateVelocities(velocities, acceleration);
}

void cpu_computeCenterOfMass(glm::vec4* bodies)
{
    glm::vec4 res(0.0f,0.0f,0.0f,0.0f);
    for(int i = 0; i < AbstractSimulation::N; ++i)
    {
        glm::vec4 tmp(bodies[i].x * bodies[i].w,
                      bodies[i].y * bodies[i].w,
                      bodies[i].z * bodies[i].w, bodies[i].w);
        res += tmp;
    }
    glm::vec3 center(res.x / res.w,res.y / res.w,res.z / res.w);
    AbstractSimulation::centerOfMass = center;
}
