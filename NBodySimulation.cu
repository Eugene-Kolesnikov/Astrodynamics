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
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dev_bodies, &num_bytes, cuda_pos_resource));
    cu_loadInitParameters(dev_bodies, N_Bodies, num_bytes);
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

    glm::mat4 Projection = glm::perspective(45.5f, (float)Registry::width / Registry::height, 0.1f, 100.0f);
    glm::mat4 PV = Projection * glm::lookAt(Registry::centerOfMass + sphericalToCartesian(Registry::cameraPos),
                                            Registry::centerOfMass,
                                            Registry::upVector);
    glUseProgram(glProgram);

	GLint PVM = glGetUniformLocation(glProgram, "PVM");
	glUniformMatrix4fv(PVM, 1, GL_FALSE, glm::value_ptr(PV));

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pos_resource, 0));
    cu_shiftParameters(dev_bodies, N);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pos_resource, 0));
	cudaDeviceSynchronize();

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
    for(int i = 0; i < N; ++i) {
        glm::vec3 pos = glm::ballRand(1.0f); // power 10^12 meters
        float mass = glm::ballRand(1.0f).x; // power 10^20 kilograms
        // => gamma should be smaller by 16 powers
        N_Bodies[i] = glm::vec4(pos.x, pos.y, pos.z, mass);
    }
}
