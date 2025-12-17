#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 v_uv;
layout (location = 3) in vec3 f_color;
layout (location = 4) in vec4 f_light_space_pos;

layout (location = 0) out vec4 final_color;

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

layout(std430, binding = 2) readonly buffer PointLightsBlock {
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

layout(std430, binding = 3) readonly buffer SpotLightsBlock {
    int numSpotLights;
    int spad0;
    int spad1;
    int spad2;
    SpotLight spotLights[];
};

// Shadow sampler
layout (binding = 4) uniform sampler2DShadow shadow_map;

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

/*
  computeShadow:
  - использует f_light_space_pos (передано из вершинного)
  - делает perspective divide -> [0,1]
  - применяет PCF 3x3 с использованием shadow_map_texel_size
  - возвращает значение shadow в диапазоне [0,1], где 0 = освещено, 1 = полностью в тени
*/
float computeShadow(vec3 normal, vec3 light_dir) {
    vec3 proj_coords = f_light_space_pos.xyz / f_light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;

    // вне карты — считаем освещенной
    if (proj_coords.z > 1.0 || proj_coords.z < 0.0 ||
        proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 0.0;
    }

    float ndotl = max(dot(normal, light_dir), 0.0);
    float bias = max(shadow_bias * (1.0 - ndotl), shadow_bias * 0.5);

    vec2 texel_size = vec2(shadow_map_texel_size);

    float lit = 0.0;
    float samples = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 offset = vec2(x, y) * texel_size;
            // sampler2DShadow делает сравнение depth <= sampleDepth
            lit += texture(shadow_map, vec3(proj_coords.xy + offset, proj_coords.z - bias));
            samples += 1.0;
        }
    }

    float visible = lit / samples; // fraction of samples that are lit (1.0 = fully lit)
    float shadow = 1.0 - visible;  // 1.0 = fully shadowed
    return clamp(shadow, 0.0, 1.0);
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
    if (dir_light_enabled > 0.5) {
        vec3 Ldir = -normalize(light_direction);
        float intensity = dir_light_color.w;
        vec3 lightCol = dir_light_color.xyz;

        // compute shadow only for surfaces facing the light
        float diffN = max(dot(N, Ldir), 0.0);

        float shadow = 0.0;
        if (diffN > 0.0) {
            shadow = computeShadow(N, Ldir) * shadow_strength;
        }

        color += blinnPhong(N, V, Ldir, lightCol, intensity, baseAlbedo, specColor, shininess, specularScale, texSpec) * (1.0 - shadow);
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
