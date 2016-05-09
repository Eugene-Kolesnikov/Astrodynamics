#include "abstractsimulation.hpp"
#include <fstream>
#include <sstream>
#include <vector>

glm::vec2 AbstractSimulation::pMouse = glm::vec2(0.0f, 0.0f);
glm::vec3 AbstractSimulation::cameraPos = glm::vec3(1200.0f, M_PI_2, 0.0f); // in spherical coordinates (r, theta, phi)
glm::vec3 AbstractSimulation::centerOfMass = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 AbstractSimulation::upVector = glm::vec3(0.0f, 0.0f, 1.0f);

unsigned int AbstractSimulation::N = 0;

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
    potentialFieldRendering = false;
}

AbstractSimulation::~AbstractSimulation()
{
    delete bodies;
    delete velocities;
}

void AbstractSimulation::setPotentialFieldRendering(bool enable)
{
    potentialFieldRendering = enable;
}

void AbstractSimulation::init(int dimensions)
{
    if(potentialFieldRendering == true) {
        AbstractSimulation::N = 4*1024;
    } else {
        AbstractSimulation::N = 11*1024;
    }

    initNBodyPositions(_2D_SIMULATION_);

    size_t num_bytes = AbstractSimulation::N * sizeof(glm::vec4);

    // Create vertex array
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

    // Create a buffer of bodies' positions and allocate memory
	glGenBuffers(1, &posBodiesBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, posBodiesBuffer);
	glBufferData(GL_ARRAY_BUFFER, num_bytes, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if(potentialFieldRendering == true) {
        // Fragmentation of a potential field
        potential_Hx = 125; // x-axis
        potential_Hy = 125; // y-axis

        // Create a buffer of potential field positions
        createPotentialPosition_VBO(&potentialFieldPositionBuffer, potential_Hx, potential_Hy);

        // Create a buffer of potential field colors
        createPotentialColor_VBO(&potentialFieldColorBuffer, potential_Hx, potential_Hy);

        // Create a buffer of potential field indeces
    	createPotential_IBO(&potentialFieldIndexBuffer, potential_Hx, potential_Hy);
    }

    glBindVertexArray(0);

    { // shaders for rendering bodies
        bodiesVertexShader = glCreateShader(GL_VERTEX_SHADER);
    	bodiesFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    	const GLchar* vShaderSource = loadFile("./shaders/nbody.vert.glsl");
    	const GLchar* fShaderSource = loadFile("./shaders/nbody.frag.glsl");
    	glShaderSource(bodiesVertexShader, 1, &vShaderSource, NULL);
    	glShaderSource(bodiesFragmentShader, 1, &fShaderSource, NULL);
    	delete [] vShaderSource;
    	delete [] fShaderSource;
    	glCompileShader(bodiesVertexShader);
    	glCompileShader(bodiesFragmentShader);
    	bodiesShaderProgram = glCreateProgram();
    	glAttachShader(bodiesShaderProgram, bodiesVertexShader);
    	glAttachShader(bodiesShaderProgram, bodiesFragmentShader);
    	glLinkProgram(bodiesShaderProgram);
    }

    if(potentialFieldRendering == true) { // shaders for rendering potential field
        potentialVertexShader = glCreateShader(GL_VERTEX_SHADER);
    	potentialFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    	const GLchar* vShaderSource = loadFile("./shaders/potential.vert.glsl");
    	const GLchar* fShaderSource = loadFile("./shaders/potential.frag.glsl");
    	glShaderSource(potentialVertexShader, 1, &vShaderSource, NULL);
    	glShaderSource(potentialFragmentShader, 1, &fShaderSource, NULL);
    	delete [] vShaderSource;
    	delete [] fShaderSource;
    	glCompileShader(potentialVertexShader);
    	glCompileShader(potentialFragmentShader);
    	potentialShaderProgram = glCreateProgram();
    	glAttachShader(potentialShaderProgram, potentialVertexShader);
    	glAttachShader(potentialShaderProgram, potentialFragmentShader);
    	glLinkProgram(potentialShaderProgram);
    }
}

void AbstractSimulation::initNBodyPositions(int dimensions)
{
    bodies = new glm::vec4[AbstractSimulation::N];
    velocities = new glm::vec3[AbstractSimulation::N];
    for(int i = 1; i < AbstractSimulation::N; ++i) {
        glm::vec3 pos = glm::ballRand(1000.0f);
        float mass;
        if(potentialFieldRendering == true) {
            mass = 1000 + fabs(glm::ballRand(1000.0f).x);
        } else {
            mass = 100 + fabs(glm::ballRand(100.0f).x);
        }
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
    }
    if(potentialFieldRendering == true) {
        bodies[0] = glm::vec4(0.0f,0.0f,0.0f,400000.0f); velocities[0] = glm::vec3(0.0f,0.0f,0.0f);
    } else {
        bodies[0] = glm::vec4(0.0f,0.0f,0.0f,2000000.0f); velocities[0] = glm::vec3(0.0f,0.0f,0.0f);
    }
}

void AbstractSimulation::createPotentialPosition_VBO(GLuint *id, int w, int h)
{
    float radius = 1250.0f;
    potentialFieldPositions = new glm::vec4[w * h];
    glm::vec3* pos = new glm::vec3[w * h];
    glGenBuffers(1, id);
	glBindBuffer(GL_ARRAY_BUFFER, *id);
    for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
            pos[y*w+x] = glm::vec3(-20.0f, -radius + 2*radius / w * x, -radius + 2*radius / h * y);
            potentialFieldPositions[y*w+x] =
                glm::vec4(0.0f, -radius + 2*radius / w * x, -radius + 2*radius / h * y, 1.0f);
	    }
	}
	glBufferData(GL_ARRAY_BUFFER, w * h * sizeof(glm::vec3), pos, GL_STATIC_DRAW);
	if (!pos) {
		printf("Error: createMeshPositionVBO\n");
		return;
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
    delete pos;
}

void AbstractSimulation::createPotentialColor_VBO(GLuint *id, int w, int h)
{
    glGenBuffers(1, id);
	glBindBuffer(GL_ARRAY_BUFFER, *id);
	glBufferData(GL_ARRAY_BUFFER, w * h * sizeof(glm::vec3), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void AbstractSimulation::createPotential_IBO(GLuint *id, int w, int h)
{
    std::vector<GLuint> ids;
    for (int y = 0; y < h - 1; ++y) {
        for (int x = 0; x < w - 1; ++x) {
            ids.push_back(y * w + x); ids.push_back(y * w + x+1); ids.push_back((y+1) * w + x);
            ids.push_back(y * w + x+1); ids.push_back((y+1) * w + x); ids.push_back((y+1) * w + x+1);
        }
    }

	// create index buffer
	glGenBuffers(1, id);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, ids.size() * sizeof(GLuint), ids.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
