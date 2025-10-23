#version 450

// NOTE: Attributes must match the declaration of VkVertexInputAttribute array
layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_color;

// NOTE: Must match declaration order of a C struct
layout (push_constant, std430) uniform ShaderConstants {
	mat4 mvp;
	vec3 color;
};

layout(location = 0) out vec3 f_color; // передаём во фрагментный шейдер

void main() {
    gl_Position = vec4(v_position, 1.0) * mvp;
    f_color = v_color;
}