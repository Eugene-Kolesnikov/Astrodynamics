#include "cpu_simulation.hpp"
#include "cpu_computations.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

CPU_Simulation::CPU_Simulation()
{

}

CPU_Simulation::~CPU_Simulation()
{

}

void CPU_Simulation::init(int dimensions)
{
    AbstractSimulation::init(dimensions);
    acceleration = new glm::vec3[AbstractSimulation::N];
    cpu_initVelocities(bodies, velocities, acceleration);
}

void CPU_Simulation::render()
{
    size_t num_bytes = AbstractSimulation::N * sizeof(glm::vec4);

    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);

    // actual calculations
    cpu_integrateSystem(bodies, velocities, acceleration);
    cpu_computeCenterOfMass(bodies);

    // send data to the GPU
    glBufferData(GL_ARRAY_BUFFER, num_bytes, bodies, GL_STATIC_DRAW);

    glm::mat4 Projection = glm::perspective(45.5f, (float)AbstractSimulation::width / AbstractSimulation::height, 0.0001f, 100000.0f);
    glm::mat4 PV = Projection * glm::lookAt(AbstractSimulation::centerOfMass + sphericalToCartesian(AbstractSimulation::cameraPos),
                                            AbstractSimulation::centerOfMass,
                                            AbstractSimulation::upVector);
    glUseProgram(bodiesShaderProgram);

	GLint PVM = glGetUniformLocation(bodiesShaderProgram, "PVM");
	glUniformMatrix4fv(PVM, 1, GL_FALSE, glm::value_ptr(PV));

    GLuint pos = glGetAttribLocation(bodiesShaderProgram, "pos");
    glVertexAttribPointer(pos, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(pos);

    glPointSize(4);
    glDrawArrays(GL_POINTS, 0,AbstractSimulation::N);
    glFlush();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

void CPU_Simulation::integrateSystem(glm::vec4* bodies, glm::vec3* velocities, glm::vec3* acceleration, unsigned int N)
{

}

void CPU_Simulation::computeCenterOfMass(glm::vec4* bodies, unsigned int N)
{

}
