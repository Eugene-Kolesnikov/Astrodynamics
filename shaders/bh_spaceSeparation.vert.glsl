#version 410 core

layout(location = 0) in vec3 pos;

uniform mat4 PVM;

out vec4 position;

void main() {
    position = PVM * vec4(pos.xyz, 1.0);
    gl_Position = position;
}
