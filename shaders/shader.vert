#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec3 v_color;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 v_uv;
layout (location = 3) out vec3 f_color;
layout (location = 4) out vec4 f_light_space_pos;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	mat4 light_view_projection;
	vec4 camera_position;
	vec3 light_position;
	float _pad0;
	vec3 light_direction;
	float dir_light_enabled;
	vec4 dir_light_color;
	float time;
	float shadow_bias;
	float shadow_strength;
	float shadow_map_texel_size;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec4 albedo_color;
	vec4 specular_color;
	vec4 misc;
	vec4 uv_scale_offset; // xy scale, zw offset
};

void main() {
	vec4 position = model * vec4(v_position, 1.0f);
	vec4 normal = model * vec4(v_normal, 0.0f);
	vec3 world_normal = normalize(normal.xyz);

	gl_Position = view_projection * position;

	f_position = position.xyz;
	f_normal = world_normal;
	vec2 scaledUV = f_uv * uv_scale_offset.xy + uv_scale_offset.zw;
	v_uv = scaledUV;
	f_color = v_color;

	// позиция в пространстве источника света (для shadow sampling)
	f_light_space_pos = light_view_projection * position;
}
