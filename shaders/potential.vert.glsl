#version 410 core

layout(location = 0) in vec4 pos;
layout(location = 1) in vec3 col;

uniform mat4 PVM;

out vec3 color;

void main() {
    vec4 position = PVM * vec4(pos.xyz, 1.0);
    gl_Position = position;
    color = col;
}
