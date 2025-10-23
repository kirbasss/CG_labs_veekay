#version 450

// NOTE: out attributes of vertex shader must be in's
layout(location = 0) in vec3 f_color;

// NOTE: Pixel color
layout (location = 0) out vec4 final_color;

// NOTE: Must match declaration order of a C struct
layout (push_constant, std430) uniform ShaderConstants {
	mat4 mvp;
	vec3 color;
};

void main() {
	final_color = vec4(f_color * color, 1.0f);
}
