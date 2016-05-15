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
    cpu_l = new float[AbstractSimulation::N];
    cpu_pivots = new glm::vec3[AbstractSimulation::N];

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

    // Create a buffer of lines' positions and allocate memory
    size_t lines_bytes = 2*AbstractSimulation::N * sizeof(glm::vec3);
	glGenBuffers(1, &linesBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, linesBuffer);
	glBufferData(GL_ARRAY_BUFFER, lines_bytes, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    BH_spaceSeparationVertexShader = glCreateShader(GL_VERTEX_SHADER);
    BH_spaceSeparationFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar* vShaderSource = loadFile("./shaders/bh_spaceSeparation.vert.glsl");
    const GLchar* fShaderSource = loadFile("./shaders/bh_spaceSeparation.frag.glsl");
    glShaderSource(BH_spaceSeparationVertexShader, 1, &vShaderSource, NULL);
    glShaderSource(BH_spaceSeparationFragmentShader, 1, &fShaderSource, NULL);
    delete [] vShaderSource;
    delete [] fShaderSource;
    glCompileShader(BH_spaceSeparationVertexShader);
    glCompileShader(BH_spaceSeparationFragmentShader);
    BH_spaceSeparationShaderProgram = glCreateProgram();
    glAttachShader(BH_spaceSeparationShaderProgram, BH_spaceSeparationVertexShader);
    glAttachShader(BH_spaceSeparationShaderProgram, BH_spaceSeparationFragmentShader);
    glLinkProgram(BH_spaceSeparationShaderProgram);
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

    {
        glUseProgram(BH_spaceSeparationShaderProgram);
        GLint PVM = glGetUniformLocation(BH_spaceSeparationShaderProgram, "PVM");
    	glUniformMatrix4fv(PVM, 1, GL_FALSE, glm::value_ptr(PV));

        glBindBuffer(GL_ARRAY_BUFFER, linesBuffer);
        updateLines();
        glBufferData(GL_ARRAY_BUFFER, 2*AbstractSimulation::N*sizeof(glm::vec3), bh_lines, GL_DYNAMIC_DRAW);
        GLuint pos = glGetAttribLocation(BH_spaceSeparationShaderProgram, "pos");
        glVertexAttribPointer(pos, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    	glEnableVertexAttribArray(pos);

        glDrawArrays(GL_LINES, 0, 2*AbstractSimulation::N);
    }



    {
        glUseProgram(bodiesShaderProgram);

    	GLint PVM = glGetUniformLocation(bodiesShaderProgram, "PVM");
    	glUniformMatrix4fv(PVM, 1, GL_FALSE, glm::value_ptr(PV));

        glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);
        GLuint pos = glGetAttribLocation(bodiesShaderProgram, "pos");
        glVertexAttribPointer(pos, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
    	glEnableVertexAttribArray(pos);

        glPointSize(4);
        glDrawArrays(GL_POINTS, 0, AbstractSimulation::N);
    }

    glFlush();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

void GPU_BarnesHut_Simulation::updateLines()
{
    checkCudaErrors(cudaMemcpy(cpu_l, dev_l, AbstractSimulation::N * sizeof(float), cudaMemcpyDeviceToHost ));
    checkCudaErrors(cudaMemcpy(cpu_pivots, dev_pivots, AbstractSimulation::N * sizeof(glm::vec3), cudaMemcpyDeviceToHost ));
    for(int i = AbstractSimulation::N-1, j = 0; i > 0; --i, j+=2)
    {
        if(j == 0) {
            glm::vec3 ul(0.0,cpu_pivots[i].y+cpu_l[i],cpu_pivots[i].z);
            glm::vec3 ur(0.0,cpu_pivots[i].y+cpu_l[i],cpu_pivots[i].z+cpu_l[i]);
            glm::vec3 bl(0.0,cpu_pivots[i].y,cpu_pivots[i].z);
            glm::vec3 br(0.0,cpu_pivots[i].y,cpu_pivots[i].z+cpu_l[i]);
            bh_lines[j+0] = ul; bh_lines[j+1] = ur;
            bh_lines[j+2] = ur; bh_lines[j+3] = br;
            bh_lines[j+4] = br; bh_lines[j+5] = bl;
            bh_lines[j+6] = bl; bh_lines[j+7] = ul;
            j += 8;

        }
        glm::vec3 l(0.0,cpu_pivots[i].y+cpu_l[i]/2.0f,cpu_pivots[i].z);
        glm::vec3 r(0.0,cpu_pivots[i].y+cpu_l[i]/2.0f,cpu_pivots[i].z+cpu_l[i]);
        glm::vec3 u(0.0,cpu_pivots[i].y+cpu_l[i],cpu_pivots[i].z+cpu_l[i]/2.0f);
        glm::vec3 d(0.0,cpu_pivots[i].y,cpu_pivots[i].z+cpu_l[i]+cpu_l[i]/2.0f);
        bh_lines[j+0] = l; bh_lines[j+1] = r;
        bh_lines[j+0] = u; bh_lines[j+1] = d;
    }
}
