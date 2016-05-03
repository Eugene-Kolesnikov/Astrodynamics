#include "NBodySimulation.hpp"
#include <glm/gtc/random.hpp>
#include <iostream>
#include <cmath>

#define N 1024

NBodySimulation::NBodySimulation()
{
}

NBodySimulation::~NBodySimulation()
{
    delete N_Bodies;
}

void NBodySimulation::init()
{
    //glClearColor(.0f, .0f, .0f, 1.0f);
    //glEnable(GL_DEPTH_TEST);
    //glDepthFunc(GL_LEQUAL);
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    initNBodyPositions();
    size_t num_bytes = N * sizeof(glm::vec4);

    // Create vertex array
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

    // Create buffer of verticies
	glGenBuffers(1, &posBodiesBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);
	glBufferData(GL_ARRAY_BUFFER, num_bytes, 0, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pos_resource, posBodiesBuffer, cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pos_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dev_bodies, &num_bytes, cuda_pos_resource));
    cu_loadInitParameters(dev_bodies, N_Bodies, num_bytes);
    cu_initVelocities(dev_bodies, dev_velocities, dev_acceleration, N);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0));
    cudaDeviceSynchronize();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glShaderV = glCreateShader(GL_VERTEX_SHADER);
	glShaderF = glCreateShader(GL_FRAGMENT_SHADER);
	const GLchar* vShaderSource = loadFile("nbody.vert.glsl");
	const GLchar* fShaderSource = loadFile("nbody.frag.glsl");
	glShaderSource(glShaderV, 1, &vShaderSource, NULL);
	glShaderSource(glShaderF, 1, &fShaderSource, NULL);
	delete [] vShaderSource;
	delete [] fShaderSource;
	glCompileShader(glShaderV);
	glCompileShader(glShaderF);
	glProgram = glCreateProgram();
	glAttachShader(glProgram, glShaderV);
	glAttachShader(glProgram, glShaderF);
	glLinkProgram(glProgram);
}

void NBodySimulation::render()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pos_resource, 0));
    // the actual computations
    cu_integrateSystem(dev_bodies, dev_velocities, dev_acceleration, N);
    cu_computeCenterOfMass(dev_bodies, dev_tmp_bodies, N);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0));
	cudaDeviceSynchronize();

    glm::mat4 Projection = glm::perspective(45.5f, (float)Registry::width / Registry::height, 0.0001f, 100.0f);
    glm::mat4 PV = Projection * glm::lookAt(Registry::centerOfMass + sphericalToCartesian(Registry::cameraPos),
                                            Registry::centerOfMass,
                                            Registry::upVector);
    glUseProgram(glProgram);

	GLint PVM = glGetUniformLocation(glProgram, "PVM");
	glUniformMatrix4fv(PVM, 1, GL_FALSE, glm::value_ptr(PV));

    GLuint pos = glGetAttribLocation(glProgram, "pos");
    glVertexAttribPointer(pos, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(pos);

    glPointSize(7);
    glDrawArrays(GL_POINTS, 0, N);
    glFlush();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
}

void NBodySimulation::initNBodyPositions()
{
    N_Bodies = new glm::vec4[N];
    glm::vec3* velocities = new glm::vec3[N];
    for(int i = 0; i < N; ++i) {
        glm::vec3 pos = glm::ballRand(1.0f);
        float mass = fabs(glm::ballRand(1.0f).x);
        //float mass = 0.01f; // power 10^20 kilograms
        N_Bodies[i] = glm::vec4(pos.x, pos.y, pos.z, mass);
        velocities[i] = glm::ballRand(55.0f);
        //velocities[i] = glm::vec3(0.0f, 0.0f, 100.0f);
    }
    checkCudaErrors(cudaMalloc((void**)&dev_tmp_bodies, N * sizeof(float4)));
    checkCudaErrors(cudaMalloc((void**)&dev_acceleration, N * sizeof(float3)));
    checkCudaErrors(cudaMalloc((void**)&dev_velocities, N * sizeof(float3)));
    checkCudaErrors( cudaMemcpy( dev_velocities, velocities, N * sizeof(glm::vec3), cudaMemcpyHostToDevice ) );
    delete velocities;
}
