#include "gpu_bh_simulation.hpp"
#include "gpu_bh_computations.hpp"
#include <iostream>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

glm::vec3 GPU_BarnesHut_Simulation::pivot = glm::vec3(0.0f, 0.0f, 0.0f);

GPU_BarnesHut_Simulation::GPU_BarnesHut_Simulation()
{

}

GPU_BarnesHut_Simulation::~GPU_BarnesHut_Simulation()
{
    cudaDeviceReset();
}

void GPU_BarnesHut_Simulation::init(int dimensions)
{
    AbstractSimulation::init(dimensions);

    checkCudaErrors(cudaMalloc((void**)&dev_acceleration, AbstractSimulation::N * sizeof(float3)));
    checkCudaErrors(cudaMalloc((void**)&dev_velocities, AbstractSimulation::N * sizeof(float3)));
    checkCudaErrors(cudaMemcpy(dev_velocities, velocities, AbstractSimulation::N * sizeof(glm::vec3), cudaMemcpyHostToDevice ));
    checkCudaErrors(cudaMalloc((void**)&dev_tmp_bodies, AbstractSimulation::N * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void**)&dev_l, AbstractSimulation::N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&dev_delta, AbstractSimulation::N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&dev_pivots, AbstractSimulation::N * sizeof(float3)));
    checkCudaErrors(cudaMalloc((void**)&dev_child, AbstractSimulation::N * sizeof(int4)));
    checkCudaErrors(cudaMalloc((void**)&dev_nextCell, sizeof(int)));

    size_t num_bytes = AbstractSimulation::N * sizeof(glm::vec4);
    size_t num_bytes_bodies_cells = 2 * AbstractSimulation::N * sizeof(glm::vec4);

	glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);

    // register cuda resource
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pos_resource, posBodiesBuffer, cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pos_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dev_bodies_cells, &num_bytes_bodies_cells, cuda_pos_resource));
    // initialization on the GPU
    cu_BHloadInitParameters(dev_bodies_cells, bodies, num_bytes);
    cu_BHinitVelocities(dev_bodies_cells, dev_velocities, dev_acceleration, AbstractSimulation::N);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0));
    cudaDeviceSynchronize();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GPU_BarnesHut_Simulation::render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pos_resource, 0));
    // the actual computations
    /*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*/
    /*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*/
    /*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*/
    cu_BHintegrateSystem(dev_bodies_cells, // coordinates + mass (bodies + cells)
                         dev_tmp_bodies, // for the caclation of a pivot point (bodies)
                         dev_velocities, // (bodies)
                         dev_acceleration, // (bodies)
                         dev_l, // the width of a space block (cells)
                         dev_delta, // the distance between the center of mass and the geometric center (cells)
                         dev_pivots, // bottom left points which represent the begining of subspaces (cells)
                         dev_child, // children pointers (cells)
                         dev_nextCell,
                         AbstractSimulation::N); /*/////////////////////////////////////////////////////////////////////*/
    /*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*/
    /*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*/
    /*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*//*//////////////////////////////////////////////////*/
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0));
	cudaDeviceSynchronize();

    glm::mat4 Projection = glm::perspective(45.5f, (float)AbstractSimulation::width / AbstractSimulation::height, 0.0001f, 100000.0f);
    glm::mat4 PV = Projection * glm::lookAt(AbstractSimulation::centerOfMass + sphericalToCartesian(AbstractSimulation::cameraPos),
                                            AbstractSimulation::centerOfMass,
                                            AbstractSimulation::upVector);
    glUseProgram(bodiesShaderProgram);

	GLint PVM = glGetUniformLocation(bodiesShaderProgram, "PVM");
	glUniformMatrix4fv(PVM, 1, GL_FALSE, glm::value_ptr(PV));

    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);
    GLuint pos = glGetAttribLocation(bodiesShaderProgram, "pos");
    glVertexAttribPointer(pos, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(pos);

    glPointSize(4);
    glDrawArrays(GL_POINTS, 0,AbstractSimulation::N);
    glFlush();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}
