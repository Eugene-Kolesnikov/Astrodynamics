#ifndef GPU_BH_SIMULATION_H
#define GPU_BH_SIMULATION_H

#include "abstractsimulation.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class GPU_BarnesHut_Simulation : public AbstractSimulation
{
public:
    GPU_BarnesHut_Simulation();
    ~GPU_BarnesHut_Simulation();

    virtual void init(int dimensions);
    virtual void render();

protected:
    struct cudaGraphicsResource* cuda_pos_resource;
    struct cudaGraphicsResource* cudaPotentialColorResource;

protected:
    float4* dev_bodies_cells; // coordinates + mass (bodies + cells)
    float4* dev_tmp_bodies; // for the caclation of a pivot point (bodies)
    float3* dev_velocities; // (bodies)
    float3* dev_acceleration; // (bodies)
    float* dev_l; // the width of a space block (cells)
    float* dev_delta; // the distance between the center of mass and the geometric center (cells)
    float3* dev_pivots; // bottom left points which represent the begining of subspaces (cells)
    int4* dev_child; // children pointers (cells)
    int* dev_nextCell;

protected:
    float* cpu_l;
    glm::vec3* cpu_pivots;
    glm::vec3* bh_lines;

protected:
    void updateLines();

public:
    static glm::vec3 pivot; // bottom left point which represents the begining of a main (root) space

protected: // shaders
    GLuint BH_spaceSeparationVertexShader;
    GLuint BH_spaceSeparationFragmentShader;
    GLuint BH_spaceSeparationShaderProgram;

protected: // OpenGL vertex buffers
    GLuint linesBuffer;
};

#endif
