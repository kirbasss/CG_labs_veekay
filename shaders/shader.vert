#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;
layout (location = 3) in vec3 v_color;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec3 f_color;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 dir_light_dir;
	vec4 dir_light_color;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec4 albedo_color;
	vec4 specular_color;
	vec4 misc;
};

void main() {
	vec4 position = model * vec4(v_position, 1.0f);
	vec4 normal = model * vec4(v_normal, 0.0f);
	// mat3 normalMatrix = mat3(transpose(inverse(model)));
	// vec3 world_normal = normalize(normalMatrix * v_normal);
	vec3 world_normal = normalize(normal.xyz);

	gl_Position = view_projection * position;

	f_position = position.xyz;
	f_normal = world_normal.xyz;
	f_uv = v_uv;
	f_color = v_color;
}
