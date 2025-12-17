#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

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
	vec4 uv_scale_offset;
};

void main() {
    vec4 world_pos = model * vec4(v_position, 1.0);
    gl_Position = light_view_projection * world_pos;
}
