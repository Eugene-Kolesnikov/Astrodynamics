#include "NBodySimulation.hpp"
#include <glm/gtc/random.hpp>
#include <iostream>

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

    // Create buffer of verticies
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	// create vertex buffers and register with CUDA
	glGenBuffers(1, &posBodiesBuffer);
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
    glm::mat4 PV = Projection * glm::lookAt(sphericalToCartesian(Registry::cameraPos),
                                            Registry::centerOfMass,
                                            Registry::upVector);

    glUseProgram(glProgram);

	GLint PVM = glGetUniformLocation(glProgram, "PVM");
	glUniformMatrix4fv(PVM, 1, GL_FALSE, glm::value_ptr(PV));

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);

    glBufferData(GL_ARRAY_BUFFER, N * sizeof(glm::vec3), N_Bodies, GL_STATIC_DRAW);
    GLuint pos = glGetAttribLocation(glProgram, "pos");
    glVertexAttribPointer(pos, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
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
    N_Bodies = new glm::vec3[N];
    for(int i = 0; i < N; ++i) {
        N_Bodies[i] = glm::ballRand(1.0f);
    }
}

/*void computeCUDA()
{
    // update heightmap values in vertex buffer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_heightVB_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&g_hptr, &num_bytes, cuda_heightVB_resource));
	cudaUpdateHeightmapKernel(g_hptr, d_ht, meshSize, meshSize);
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_heightVB_resource, 0));

	cudaDeviceSynchronize();
}*/
