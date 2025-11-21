#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
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

vec3 blinnPhong(vec3 N, vec3 V, vec3 Ldir, vec3 lightColor, float intensity, vec3 albedo, vec3 specColor, float shininess) {
    float NdotL = max(dot(N, Ldir), 0.0);
    vec3 diffuse = albedo * NdotL;

    vec3 H = normalize(Ldir + V);
    float NdotH = max(dot(N, H), 0.0);
    float specFactor = pow(NdotH, shininess);
    vec3 specular = specColor * specFactor;
	// specular *= 1.5f;

    return (diffuse + specular) * lightColor * intensity;
}

// simple pseudo-random for lava jitter (not high quality noise, but ok)
float prand(vec2 co) {
    // classic hash
    float t = dot(co, vec2(12.9898,78.233));
    return fract(sin(t) * 43758.5453);
}

void main() {
	// pull misc
	float shininess = misc.x;
	float emissiveStrength = misc.y;
	float opacity = misc.z;
	float specScale = misc.w;

	// base normal/view
	vec3 N = normalize(f_normal);
	vec3 V = normalize(camera_position.xyz - f_position);

	// Animated UV per-material: choose behavior by material flags encoded in shininess sign or another convention:
	// We'll use conventions:
	//  - water: shininess > 0 and shininess < 1000 (normal)
	//  - lava: we will set specScale negative in C++ to indicate "lava" style.
	// To keep it robust, we'll detect lava by specScale < 0.0.
	vec2 uv = f_uv;
	bool isLavaJitter = specScale < 0.0;
	bool isWaterSmooth = !isLavaJitter;

	float t = time;

	// animated UVs
	if (isWaterSmooth) {
		// slow smooth flow sin/cos offset
		vec2 flow = vec2(sin(t * 0.35), cos(t * 0.27)) * 0.02;
		uv += flow;
	} else {
		// lava: slow drift + jittering patches that appear/disappear
		// base drift:
		uv += vec2(t * 0.05, -t * 0.02);

		// jitter: small cellular-like perturbation based on prand
		vec2 cell = floor(uv * 6.0); // cells
		float cellNoise = prand(cell + floor(t * 0.4));
		// make jitter stronger in some cells
		vec2 jitter = vec2(prand(cell + 1.23), prand(cell + 7.89)) - 0.5;
		uv += jitter * (cellNoise * 0.12);
	}

	vec4 albedoSample = texture(u_albedo, uv);
	vec3 texAlbedo = albedoSample.bgr;
	vec3 texSpec = texture(u_specular, f_uv).bgr;
	vec3 emissiveCol = texture(u_emissive, f_uv).bgr;
	float emissiveAlpha = texture(u_emissive, f_uv).a;

	vec3 albedo = texAlbedo * albedo_color.xyz * f_color;

	float specTexIntensity = (texSpec.r + texSpec.g + texSpec.b) / 3.0;
	vec3 specColor = specular_color.xyz * specTexIntensity * abs(specScale);
	float maxSpec = 1.5;
	if (isLavaJitter) {
		// for lava we want to limit shiny hotspots: reduce scale and clamp
		specColor *= 0.6;
		maxSpec = 0.6;
	}
	specColor = clamp(specColor, vec3(0.0), vec3(maxSpec));

    vec3 ambient = vec3(0.01);
    vec3 color = ambient * albedo;

	// directional light
	{
		if (dir_light_dir.w > 0.5) {
			vec3 Ldir = normalize(dir_light_dir.xyz);
			float intensity = dir_light_color.w;
			vec3 lightCol = dir_light_color.xyz;

			float NdotL = max(dot(N, Ldir), 0.0);
			vec3 H = normalize(Ldir + V);
			float NdotH = max(dot(N, H), 0.0);
			vec3 diff = albedo * NdotL;
			vec3 spec = specColor * pow(NdotH, shininess);
			color += (diff + spec) * lightCol * intensity;
		}
	}

	// point light
	for (int i = 0; i < numPointLights; ++i) {
		PointLight pl = pointLights[i];
		if (pl.atten.w < 0.5) continue;

		vec3 lp = pl.pos_int.xyz;
		float intensity = pl.pos_int.w;
		vec3 lightCol = pl.color_radius.xyz;
		vec3 attenParams = pl.atten.xyz;

		vec3 Ldir = normalize(lp - f_position);
		float att = calcAttenuation(lp, f_position, attenParams);
		float maxRadius = pl.color_radius.w;
		if (maxRadius > 0.001) {
			float d = length(lp - f_position);
			float radial = clamp(1.0 - (d / maxRadius), 0.0, 1.0);
			att *= radial;
		}

		vec3 contrib = blinnPhong(N, V, Ldir, lightCol, intensity, albedo, specColor, shininess);
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

		vec3 Ldir = normalize(lp - f_position);
		float theta = dot(-dir, Ldir);
		float spot = smoothstep(outerCos, innerCos, theta);
		float att = calcAttenuation(lp, f_position, attenParams);

		vec3 contrib = blinnPhong(N, V, Ldir, lightCol, intensity, albedo, specColor, shininess);
		color += contrib * att * spot;
	}

	// emissive contribution
	if (emissiveAlpha > 0.0001 && emissiveStrength > 0.0001) {
		vec3 e = emissiveCol * emissiveStrength * emissiveAlpha;
		color += e;
	}

	float finalAlpha = opacity * albedoSample.a;

	color = max(color, vec3(0.0));
	color = min(color, vec3(1.0));

	final_color = vec4(color, 1.0f);
}
