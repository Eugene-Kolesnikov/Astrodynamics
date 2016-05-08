#include "gpu_simulation.hpp"
#include "gpu_computations.hpp"
#include <iostream>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

GPU_Simulation::GPU_Simulation()
{
}

GPU_Simulation::~GPU_Simulation()
{
    cudaDeviceReset();
}

void GPU_Simulation::init(int dimensions)
{
    AbstractSimulation::init(dimensions);

    checkCudaErrors(cudaMalloc((void**)&dev_tmp_bodies, AbstractSimulation::N * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void**)&dev_acceleration, AbstractSimulation::N * sizeof(float3)));
    checkCudaErrors(cudaMalloc((void**)&dev_velocities, AbstractSimulation::N * sizeof(float3)));
    checkCudaErrors( cudaMemcpy( dev_velocities, velocities, AbstractSimulation::N * sizeof(glm::vec3), cudaMemcpyHostToDevice ) );

    size_t num_bytes = AbstractSimulation::N * sizeof(glm::vec4);

	glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);

    // register cuda resource
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pos_resource, posBodiesBuffer, cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pos_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dev_bodies, &num_bytes, cuda_pos_resource));
    // initialization on the GPU
    cu_loadInitParameters(dev_bodies, bodies, num_bytes);
    cu_initVelocities(dev_bodies, dev_velocities, dev_acceleration, AbstractSimulation::N);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0));
    cudaDeviceSynchronize();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GPU_Simulation::render()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pos_resource, 0));
    // the actual computations
    cu_integrateSystem(dev_bodies, dev_velocities, dev_acceleration, AbstractSimulation::N);
    cu_computeCenterOfMass(dev_bodies, dev_tmp_bodies, AbstractSimulation::N);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0));
	cudaDeviceSynchronize();

    glm::mat4 Projection = glm::perspective(45.5f, (float)AbstractSimulation::width / AbstractSimulation::height, 0.0001f, 100000.0f);
    glm::mat4 PV = Projection * glm::lookAt(AbstractSimulation::centerOfMass + sphericalToCartesian(AbstractSimulation::cameraPos),
                                            AbstractSimulation::centerOfMass,
                                            AbstractSimulation::upVector);
    glUseProgram(glProgram);

	GLint PVM = glGetUniformLocation(glProgram, "PVM");
	glUniformMatrix4fv(PVM, 1, GL_FALSE, glm::value_ptr(PV));

    GLuint pos = glGetAttribLocation(glProgram, "pos");
    glVertexAttribPointer(pos, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(pos);

    glPointSize(4);
    glDrawArrays(GL_POINTS, 0,AbstractSimulation::N);
    glFlush();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}
