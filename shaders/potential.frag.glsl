#version 410 core

in vec3 color;
out vec4 fColor;

void main (void) {
    fColor = vec4(color.xyz,1);
    //fColor = vec4(1,1,1,1);
}
