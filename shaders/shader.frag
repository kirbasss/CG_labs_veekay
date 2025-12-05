#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 v_uv;
layout (location = 3) in vec3 f_color;

layout (location = 0) out vec4 final_color;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 dir_light_dir;
	vec4 dir_light_color;
	float time;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec4 albedo_color;
	vec4 specular_color;
	vec4 misc;
};

// per-material samplers (set = 1)
layout (set = 1, binding = 0) uniform sampler2D u_albedo;
layout (set = 1, binding = 1) uniform sampler2D u_specular;
layout (set = 1, binding = 2) uniform sampler2D u_emissive;
layout (set = 1, binding = 3, std140) uniform MaterialParams {
    vec4 extra; // x = scrollEnabled, y = speed, z = dir.x, w = dir.y
};

struct PointLight {
    vec4 pos_int;
    vec4 color_radius;
    vec4 atten;
};

layout(std430, binding = 2) buffer PointLightsBlock {
    int numPointLights;
    int pad0;
    int pad1;
    int pad2;
    PointLight pointLights[];
};

struct SpotLight {
    vec4 pos_int;   
    vec4 dir_inner;    
    vec4 color_outer;  
    vec4 atten;
};

layout(std430, binding = 3) buffer SpotLightsBlock {
    int numSpotLights;
    int spad0;
    int spad1;
    int spad2;
    SpotLight spotLights[];
};

float calcAttenuation(vec3 lightPos, vec3 fragPos, vec3 attenParams) {
    float dist = length(lightPos - fragPos);
    float constant = attenParams.x;
    float linear = attenParams.y;
    float quadratic = attenParams.z;
    return 1.0 / max(0.0001, constant + linear * dist + quadratic * dist * dist);
}

vec3 blinnPhong(vec3 N, vec3 V, vec3 Ldir, vec3 lightColor, float intensity, vec3 albedo, vec3 specColor, float shininess, float specularScale, vec3 texSpec) {
    float NdotL = max(dot(N, Ldir), 0.0);
    vec3 diffuse = albedo * NdotL;

    vec3 H = normalize(Ldir + V);
    float NdotH = max(dot(N, H), 0.0);
    float specFactor = pow(NdotH, shininess);
    vec3 specular = specColor * specFactor;
	specular *= specularScale;

    // specular material
    float mask = dot(texSpec, vec3(0.333));
    specular *= mask;

    return (diffuse + specular) * lightColor * intensity;
}

void main() {
    vec3 albedo = albedo_color.xyz * f_color;
    vec3 specColor = specular_color.xyz;
    float shininess = misc.x;
    float emissiveStrength = misc.y;
    float opacity = misc.z;
    float specularScale = misc.w;

    vec3 N = normalize((mat3(model) * f_normal));
    vec3 worldPos = (model * vec4(f_position, 1.0)).xyz;
    vec3 V = normalize(camera_position.xyz - worldPos);

    // compute UV with optional continuous offset (uses sampler addressing mode for mirrored repeat)
    vec2 uv = v_uv;
    if (extra.x > 0.5) {
        vec2 dir = extra.zw;
        float speed = extra.y;
        vec2 offs = dir * (speed * time);
        uv += offs;
    }

    // sample textures from material set
    vec3 texAlbedo = texture(u_albedo, uv).bgr;
    vec3 texSpec   = texture(u_specular, uv).bgr;
    vec3 texEmiss  = texture(u_emissive, uv).bgr;

    // combine albedo
    vec3 baseAlbedo = albedo * texAlbedo;

    vec3 color = vec3(0.03) * baseAlbedo;

    // directional light
    if (dir_light_dir.w > 0.5) {
        vec3 Ldir = normalize(dir_light_dir.xyz);
        float intensity = dir_light_color.w;
        vec3 lightCol = dir_light_color.xyz;

        color += blinnPhong(N, V, Ldir, lightCol, intensity, baseAlbedo, specColor, shininess, specularScale, texSpec);
    }

    // point lights
    for (int i = 0; i < numPointLights; ++i) {
        PointLight pl = pointLights[i];
        if (pl.atten.w < 0.5) continue;

        vec3 lp = pl.pos_int.xyz;
        float intensity = pl.pos_int.w;
        vec3 lightCol = pl.color_radius.xyz;
        vec3 attenParams = pl.atten.xyz;

        vec3 Ldir = normalize(lp - worldPos);
        float att = calcAttenuation(lp, worldPos, attenParams);
        float maxRadius = pl.color_radius.w;
        if (maxRadius > 0.001) {
            float d = length(lp - worldPos);
            float radial = clamp(1.0 - (d / maxRadius), 0.0, 1.0);
            att *= radial;
        }

        vec3 contrib = blinnPhong(N, V, Ldir, lightCol, intensity, baseAlbedo, specColor, shininess, specularScale, texSpec);
        color += contrib * att;
    }

    // spot lights
    for (int i = 0; i < numSpotLights; ++i) {
        SpotLight s = spotLights[i];
        if (s.atten.w < 0.5) continue;

        vec3 lp = s.pos_int.xyz;
        float intensity = s.pos_int.w;
        vec3 lightCol = s.color_outer.xyz;
        vec3 dir = normalize(s.dir_inner.xyz);
        float innerCos = s.dir_inner.w;
        float outerCos = s.color_outer.w;
        vec3 attenParams = s.atten.xyz;

        vec3 Ldir = normalize(lp - worldPos);
        float theta = dot(-dir, Ldir);
        float spot = smoothstep(outerCos, innerCos, theta);
        float att = calcAttenuation(lp, worldPos, attenParams);

        vec3 contrib = blinnPhong(N, V, Ldir, lightCol, intensity, baseAlbedo, specColor, shininess, specularScale, texSpec);
        color += contrib * att * spot;
    }

    // emissive
    color += texEmiss * emissiveStrength;

    color = max(color, vec3(0.0));
    color = min(color, vec3(1.0));

    final_color = vec4(color, opacity);
}
