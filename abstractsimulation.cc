#include "abstractsimulation.hpp"
#include <fstream>
#include <sstream>

glm::vec2 AbstractSimulation::pMouse = glm::vec2(0.0f, 0.0f);
glm::vec3 AbstractSimulation::cameraPos = glm::vec3(1200.0f, M_PI_2, 0.0f); // in spherical coordinates (r, theta, phi)
glm::vec3 AbstractSimulation::centerOfMass = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 AbstractSimulation::upVector = glm::vec3(0.0f, 0.0f, 1.0f);

unsigned int AbstractSimulation::N = 11*1024;

int AbstractSimulation::width = 0;
int AbstractSimulation::height = 0;

char* loadFile(const char *filename) {
    char* data;
    int len;
    std::ifstream ifs(filename, std::ifstream::in);
    if(ifs.is_open() == false) {
        printf("File not open!\n");
    }
    ifs.seekg(0, std::ios::end);
    len = (int)ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data = new char[len + 1];
    ifs.read(data, len);
    data[len] = 0;
    ifs.close();
    return data;
}

glm::vec3 sphericalToCartesian(glm::vec3 vec)
{
    return glm::vec3(
        vec.x * sin(vec.y) * cos(vec.z),
        vec.x * sin(vec.y) * sin(vec.z),
        vec.x * cos(vec.y)
    );
}

AbstractSimulation::AbstractSimulation()
{

}

AbstractSimulation::~AbstractSimulation()
{
    delete bodies;
    delete velocities;
}

void AbstractSimulation::init(int dimensions)
{
    initNBodyPositions(_2D_SIMULATION_);

    size_t num_bytes = AbstractSimulation::N * sizeof(glm::vec4);

    // Create vertex array
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

    // Create buffer of verticies
	glGenBuffers(1, &posBodiesBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);
	glBufferData(GL_ARRAY_BUFFER, num_bytes, 0, GL_DYNAMIC_DRAW);
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

void AbstractSimulation::initNBodyPositions(int dimensions)
{
    bodies = new glm::vec4[AbstractSimulation::N];
    velocities = new glm::vec3[AbstractSimulation::N];
    for(int i = 1; i < AbstractSimulation::N; ++i) {
        glm::vec3 pos = glm::ballRand(1000.0f);
        float mass = 100 + fabs(glm::ballRand(100.0f).x);
        //float mass = 0.01f;
        if(dimensions == _2D_SIMULATION_) {
            if(sqrtf(pos.y*pos.y+pos.z*pos.z) < 10) {
                bodies[i] = glm::vec4(0.0f, 5*pos.y, 5*pos.z, mass);
                velocities[i] = glm::vec3(0.0f, -pos.z, pos.y);
            } else if(sqrtf(pos.y*pos.y+pos.z*pos.z) < 100) {
                bodies[i] = glm::vec4(0.0f, 5*pos.y, 5*pos.z, mass);
                velocities[i] = glm::vec3(0.0f, -pos.z, pos.y);
            } else if(sqrtf(pos.y*pos.y+pos.z*pos.z) < 200) {
                bodies[i] = glm::vec4(0.0f, 3.4*pos.y, 3.4*pos.z, mass);
                velocities[i] = glm::vec3(0.0f, -pos.z, pos.y);
            } else if(sqrtf(pos.y*pos.y+pos.z*pos.z) < 300) {
                bodies[i] = glm::vec4(0.0f, 2.3*pos.y, 2.3*pos.z, mass);
                velocities[i] = glm::vec3(0.0f, -pos.z/2, pos.y/2);
            } else if(sqrtf(pos.y*pos.y+pos.z*pos.z) < 400) {
                bodies[i] = glm::vec4(0.0f, 1.7*pos.y, 1.7*pos.z, mass);
                velocities[i] = glm::vec3(0.0f, -pos.z/2.2, pos.y/2.2);
            } else if(sqrtf(pos.y*pos.y+pos.z*pos.z) < 500) {
                bodies[i] = glm::vec4(0.0f, 1.2*pos.y, 1.2*pos.z, mass);
                velocities[i] = glm::vec3(0.0f, -pos.z/2.2, pos.y/3);
            } else {
                bodies[i] = glm::vec4(0.0f, pos.y, pos.z, mass);
                velocities[i] = glm::vec3(0.0f, -pos.z/5, pos.y/5);
            }
        } else if(dimensions == _3D_SIMULATION_) {
            glm::vec3 vel = glm::ballRand(50.0f);
            bodies[i] = glm::vec4(pos.x, pos.y, pos.z, mass);
            velocities[i] = glm::vec3(vel.x, vel.y, vel.z);
        }

        //velocities[i] = glm::vec3(0.0f, 0.0f, 100.0f);
    }
    bodies[0] = glm::vec4(0.0f,0.0f,0.0f,2000000.0f); velocities[0] = glm::vec3(0.0f,0.0f,0.0f);
    // bodies[1] = glm::vec4(0.0f,1.0f,0.0f,0.1f); velocities[1] = glm::vec3(0.0f,0.0f,1.0f);
    // bodies[2] = glm::vec4(0.0f,-1.0f,0.0f,0.001f); velocities[2] = glm::vec3(0.0f,0.0f,-1.0f);
    // bodies[3] = glm::vec4(0.0f,1.0f,1.0f,0.2f); velocities[3] = glm::vec3(0.0f,-1.0f,-0.5f);
    //checkCudaErrors(cudaMalloc((void**)&dev_tmp_bodies, N * sizeof(float4)));
    //checkCudaErrors(cudaMalloc((void**)&dev_acceleration, N * sizeof(float3)));
    //checkCudaErrors(cudaMalloc((void**)&dev_velocities, N * sizeof(float3)));
    //checkCudaErrors( cudaMemcpy( dev_velocities, velocities, N * sizeof(glm::vec3), cudaMemcpyHostToDevice ) );
    //delete velocities;
}
