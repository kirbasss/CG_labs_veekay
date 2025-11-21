#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

#ifndef M_PI
constexpr float M_PI = 3.14159265358979323846f;
#endif

namespace {

constexpr float camera_fov = 70.0f;
constexpr float camera_near_plane = 0.01f;
constexpr float camera_far_plane = 100.0f;
constexpr float camera_speed = 5.0f / 100.0f;

constexpr uint32_t MAX_MATERIALS = 8;

struct Vector {
	float x, y, z;
};
constexpr uint32_t max_models = 1024;

// std140 SceneUniforms (binding = 0 in shader)
struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::vec4 camera_position;
	veekay::vec4 dir_light_dir;
	veekay::vec4 dir_light_color;
	float time;	
};

// std140 ModelUniforms (binding = 1 in shader)
struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec4 albedo_color;
	veekay::vec4 specular_color;
	veekay::vec4 misc; // misc.x = shininess; misc.y = emissiveStrength; misc.z = opacity; misc.w = specularScale/flag
};

// std430 PointLight layout (binding = 2)
struct PointLightCPU {
	veekay::vec4 pos_int;
	veekay::vec4 color_radius;
	veekay::vec4 atten;
};

// std430 SpotLight layout (binding = 3)
struct SpotLightCPU {
	veekay::vec4 pos_int;
	veekay::vec4 dir_inner;
	veekay::vec4 color_outer;
	veekay::vec4 atten;
};

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	// NOTE: You can add more attributes
	veekay::vec3 color;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

// Lighting defaults
int num_point_lights = 0;
int num_spot_lights = 0;

// Sphere / animation globals
veekay::vec3 sphere_position = {0.0f, -4.0f, 0.0f};
float sphere_rotation = 0.0f;
veekay::vec3 sphere_color = {0.7f, 0.2f, 0.924f};
bool sphere_spin = false;
float puls_amp = 0.3f;
float puls_freq = 1.5f;
float scale_value = 1.0f;

float camera_yaw = 0.0f;
float camera_pitch = 0.0f;
float camera_distance = 18.0f;
const float move_speed = 0.1f;
const float look_sensitivity = 0.0025f;

float current_time = 0.0f;

// NOTE: Trajectory (figure inf sign) params
bool animate_path = false;
float path_amp_x = 2.0f;
float path_amp_y = 1.5f;
float path_speed = 1.7f;
veekay::vec3 path_origin = { 0.0f, 0.0f, 0.0f };
bool path_origin_initialized = false;
bool prev_animate_path = false;

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color = {1.0f, 1.0f, 1.0f};
	veekay::vec3 specular_color = {1.0f, 1.0f, 1.0f};
	float shininess = 32.0f;
	int material_id = 0;

	float emissive_strength = 0.0f;
	float opacity = 1.0f;
	float specular_scale = 1.0f;    // scale multiplier; set negative for lava-style flag if desired
};

enum class CameraViewMode {
    LookAt,
    TransformInverse
};
CameraViewMode camera_view_mode = CameraViewMode::LookAt;

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	veekay::vec3 up = {0.0f, 1.0f, 0.0f};
	veekay::vec3 look_at_center = {0.0f, 0.0f, 0.0f};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	// NOTE: View matrix of camera (inverse of a transform)
	veekay::mat4 view() const;

	// NOTE: View and projection composition
	veekay::mat4 view_projection(float aspect_ratio) const;
};

// NOTE: Scene objects
inline namespace {
	Camera camera{
		.position = {0.0f, -0.5f, -3.0f}
	};

	std::vector<Model> models;
}

// NOTE: Vulkan objects
inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;

	// VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;     // set 0
	VkDescriptorSetLayout material_set_layout = VK_NULL_HANDLE;  // set 1

	// VkDescriptorSet descriptor_set;
	VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> material_descriptor_sets;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;

	Mesh plane_mesh;
	Mesh cube_mesh;
	Mesh sphere_mesh;

	veekay::graphics::Buffer* point_lights_ssbo = nullptr;
	veekay::graphics::Buffer* spot_lights_ssbo = nullptr;

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;
	veekay::graphics::Texture* empty_specular;
	VkSampler empty_specular_sampler;
	veekay::graphics::Texture* empty_emissive;
	VkSampler empty_emissive_sampler;

	// veekay::graphics::Texture* texture;
	// VkSampler texture_sampler;
	// shared sampler used for materials
	VkSampler material_sampler = VK_NULL_HANDLE;

	// helper container for textures/materials
	struct Material {
		veekay::graphics::Texture* albedo = nullptr;
		veekay::graphics::Texture* specular = nullptr;
		veekay::graphics::Texture* emissive = nullptr;
		VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
	};
	std::vector<Material> materials;
}

static veekay::graphics::Texture* loadTextureFromPNG(VkCommandBuffer cmd, const char* path) {
	std::vector<unsigned char> image; // raw pixels RGBA
	unsigned width, height;
	unsigned error = lodepng::decode(image, width, height, path);
	if (error) {
		std::cerr << "Failed to load PNG '" << path << "' : " << error << "\n";
		return nullptr;
	}
	// convert RGBA -> BGRA uint32_t little-endian (A<<24 | B<<16 | G<<8 | R)
	std::vector<uint32_t> pixels(width * height);
	for (size_t i = 0; i < width * height; ++i) {
		uint8_t r = image[4 * i + 0];
		uint8_t g = image[4 * i + 1];
		uint8_t b = image[4 * i + 2];
		uint8_t a = image[4 * i + 3];
		pixels[i] = (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);
	}
	return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_B8G8R8A8_UNORM, pixels.data());
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
	auto t = veekay::mat4::translation(position);
	auto s = veekay::mat4::scaling(scale);
    veekay::mat4 rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
    veekay::mat4 ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
    veekay::mat4 rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);
	veekay::mat4 r = rz * ry * rx;

	return s * r * t;
}

veekay::mat4 lookAt(const veekay::vec3& eye, const veekay::vec3& center, const veekay::vec3& up) {
    veekay::vec3 f = veekay::vec3::normalized(center - eye);
    veekay::vec3 s = veekay::vec3::normalized(veekay::vec3::cross(f, up));
    veekay::vec3 u = veekay::vec3::cross(s, f);

    veekay::mat4 result = veekay::mat4::identity();

    result.columns[0] = veekay::vec4{ s.x, u.x, -f.x, 0.0f };
    result.columns[1] = veekay::vec4{ s.y, u.y, -f.y, 0.0f };
    result.columns[2] = veekay::vec4{ s.z, u.z, -f.z, 0.0f };
    result.columns[3] = veekay::vec4{
        -veekay::vec3::dot(s, eye),
        -veekay::vec3::dot(u, eye),
         veekay::vec3::dot(f, eye),
         1.0f
    };

    return result;
}

veekay::mat4 Camera::view() const {
	float cp = cosf(rotation.x);
	veekay::vec3 forward = { -cp * sinf(rotation.y), sinf(rotation.x), -cp * cosf(rotation.y) };
    if (camera_view_mode == CameraViewMode::LookAt) {
        veekay::vec3 center = position + forward;
        return lookAt(position, center, up);
    } else {
        // TransformInverse: строим матрицу трансформации камеры и инвертируем

        veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(forward, up));
        veekay::vec3 camUp = veekay::vec3::cross(right, forward);

        veekay::mat4 r = veekay::mat4::identity();
        r.columns[0] = veekay::vec4{ right.x, right.y, right.z, 0.0f };
        r.columns[1] = veekay::vec4{ camUp.x, camUp.y, camUp.z, 0.0f };
        r.columns[2] = veekay::vec4{ -forward.x, -forward.y, -forward.z, 0.0f };
        r.columns[3] = veekay::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };		

        auto t = veekay::mat4::translation(position);
        auto Mc = r * t;
        return veekay::mat4::inverse(Mc);
    }
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
	return view() * projection;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
        return VK_NULL_HANDLE;
    }
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
			{
				.location = 3,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, color),
			},
		};

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.blendEnable = VK_TRUE,
			.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
			.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			.colorBlendOp = VK_BLEND_OP_ADD,
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
			.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			.alphaBlendOp = VK_BLEND_OP_ADD,
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
							VK_COLOR_COMPONENT_G_BIT |
							VK_COLOR_COMPONENT_B_BIT |
							VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = (uint32_t)(MAX_MATERIALS * 3), // albedo/spec/emissive per material
				},
				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 8,
				}
			};
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1 + MAX_MATERIALS, // scene set + material sets
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Descriptor set layout for scene (set 0)
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 2, 
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 
					.descriptorCount = 1,
			  		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
				},
				{
					.binding = 3, 
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 
					.descriptorCount = 1,
			  		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
				}
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &scene_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		
		// Descriptor set layout for materials (set 1) - three combined samplers: albedo, spec, emissive
		{
			VkDescriptorSetLayoutBinding mbindings[] = {
				{ 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }, // albedo
				{ 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }, // specular
				{ 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }, // emissive
			};
			VkDescriptorSetLayoutCreateInfo info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = (uint32_t)(sizeof(mbindings)/sizeof(mbindings[0])),
				.pBindings = mbindings };
				if (vkCreateDescriptorSetLayout(device, &info, nullptr, &material_set_layout) != VK_SUCCESS) {
					std::cerr << "Failed to create material descriptor set layout\n";
					veekay::app.running = false;
					return;
				}
			}

		// NOTE: Declare external data sources, only push constants this time
		// set 0 = scene, set 1 = material
		{
			VkDescriptorSetLayout setLayouts[] = { scene_set_layout, material_set_layout };
			VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 2,
				.pSetLayouts = setLayouts,
			};

			// NOTE: Create pipeline layout
			if (vkCreatePipelineLayout(device, &layout_info,
									nullptr, &pipeline_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan pipeline layout\n";
				veekay::app.running = false;
				return;
			}
		}

		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	// allocate scene descriptor set (set 0)
	{
		VkDescriptorSetAllocateInfo allocInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &scene_set_layout,
		};
		if (vkAllocateDescriptorSets(device, &allocInfo, &scene_descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate scene descriptor set\n";
			veekay::app.running = false;
			return;
		}
	}

	// allocate material descriptor sets (set 1) - MAX_MATERIALS
	{
		material_descriptor_sets.resize(MAX_MATERIALS);
		std::vector<VkDescriptorSetLayout> matLayouts(MAX_MATERIALS, material_set_layout);
		VkDescriptorSetAllocateInfo allocInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = (uint32_t)matLayouts.size(),
			.pSetLayouts = matLayouts.data()
		};
		if (vkAllocateDescriptorSets(device, &allocInfo, material_descriptor_sets.data()) != VK_SUCCESS) {
			std::cerr << "Failed to allocate material descriptor sets\n";
			veekay::app.running = false;
			return;
		}
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	
	const uint32_t max_point = 8;
	const uint32_t max_spot = 4;
	size_t point_ssbo_size = sizeof(int) * 4 + max_point * sizeof(PointLightCPU); // headroom for num + pads
	size_t spot_ssbo_size = sizeof(int) * 4 + max_spot * sizeof(SpotLightCPU);

	point_lights_ssbo = new veekay::graphics::Buffer(point_ssbo_size, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	spot_lights_ssbo = new veekay::graphics::Buffer(spot_ssbo_size, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	// NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
										
		// empty specular: black 2x2 (no specular contribution)
		if (vkCreateSampler(device, &info, nullptr, &empty_specular_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t emptySpecPixels[] = {
			0xFF000000, 0xFF000000,
			0xFF000000, 0xFF000000,
		};
		empty_specular = new veekay::graphics::Texture(cmd, 2, 2,
			VK_FORMAT_B8G8R8A8_UNORM, emptySpecPixels);

		// empty emissive: fully transparent
		if (vkCreateSampler(device, &info, nullptr, &empty_emissive_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t emptyEmissivePixels[] = {
			0x00000000, 0x00000000,
			0x00000000, 0x00000000,
		};
		empty_emissive = new veekay::graphics::Texture(cmd, 2, 2,
			VK_FORMAT_B8G8R8A8_UNORM, emptyEmissivePixels);
	}

	// shared material sampler (can be configured)
	{
		VkSamplerCreateInfo sInfo{};
		sInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		sInfo.magFilter = VK_FILTER_LINEAR;
		sInfo.minFilter = VK_FILTER_LINEAR;
		sInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		if (vkCreateSampler(device, &sInfo, nullptr, &material_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create material sampler\n";
			veekay::app.running = false;
			return;
		}
	}

	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
			{
				.buffer = point_lights_ssbo->buffer,
				.offset = 0,
				.range = VK_WHOLE_SIZE,
			},
			{
				.buffer = spot_lights_ssbo->buffer,
				.offset = 0,
				.range = VK_WHOLE_SIZE,
			}
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = scene_descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = scene_descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = scene_descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = scene_descriptor_set,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[3],
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	// Load a few example textures and create simple materials
	materials.resize(MAX_MATERIALS);
	// create three example materials
	{
		// material 0 - obsidian (albedo only)
		materials[0].albedo = loadTextureFromPNG(cmd, "./assets/obsidian.png");
		if (!materials[0].albedo) materials[0].albedo = missing_texture;
		materials[0].specular = empty_specular;
		materials[0].emissive = empty_emissive;

		// material 1 - lava (albedo + emissive)
		materials[1].albedo = loadTextureFromPNG(cmd, "./assets/lava.png");
		if (!materials[1].albedo) materials[1].albedo = missing_texture;
		materials[1].specular = loadTextureFromPNG(cmd, "./assets/lava_spec.png");
		if (!materials[1].specular) materials[1].specular = empty_specular;
		materials[1].emissive = loadTextureFromPNG(cmd, "./assets/lava_emissive.png");
		if (!materials[1].emissive) materials[1].emissive = empty_emissive;

		// material 2 - water (albedo + specular)
		materials[2].albedo = loadTextureFromPNG(cmd, "./assets/water.png");
		if (!materials[2].albedo) materials[2].albedo = missing_texture;
		materials[2].specular = loadTextureFromPNG(cmd, "./assets/water_spec.png");
		if (!materials[2].specular) materials[2].specular = empty_specular;
		materials[2].emissive = empty_emissive;

		// ... other materials left as default missing_texture
		for (uint32_t i = 3; i < MAX_MATERIALS; ++i) {
			materials[i].albedo = missing_texture;
			materials[i].specular = empty_specular;
			materials[i].emissive = empty_emissive;
		}
	}

	// Update each material's descriptor set (set = 1)
	for (uint32_t i = 0; i < MAX_MATERIALS; ++i) {
		VkDescriptorImageInfo albedoInfo{ .sampler = material_sampler, .imageView = materials[i].albedo->view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		VkDescriptorImageInfo specInfo{ .sampler = material_sampler, .imageView = materials[i].specular->view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		VkDescriptorImageInfo emissiveInfo{ .sampler = material_sampler, .imageView = materials[i].emissive->view, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

		VkWriteDescriptorSet writes[3]{};
		writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].dstSet = material_descriptor_sets[i];
		writes[0].dstBinding = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writes[0].pImageInfo = &albedoInfo;

		writes[1] = writes[0];
		writes[1].dstBinding = 1;
		writes[1].pImageInfo = &specInfo;

		writes[2] = writes[0];
		writes[2].dstBinding = 2;
		writes[2].pImageInfo = &emissiveInfo;

		vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

		// store descriptor_set in our helper list too
		materials[i].descriptor_set = material_descriptor_sets[i];
	}

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f,1.0f,1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f,1.0f,1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0
		};

		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Cube mesh initialization
	{
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}, {1.0f,1.0f,1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}, {1.0f,1.0f,1.0f}},

			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}, {1.0f,1.0f,1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f,1.0f,1.0f}},

			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f,1.0f,1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}, {1.0f,1.0f,1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}, {1.0f,1.0f,1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f,1.0f,1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f,1.0f,1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f,1.0f,1.0f}},

			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f,1.0f,1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f,1.0f,1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f,1.0f,1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
			8, 9, 10, 10, 11, 8,
			12, 13, 14, 14, 15, 12,
			16, 17, 18, 18, 19, 16,
			20, 21, 22, 22, 23, 20,
		};

		cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		cube_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Sphere mesh initialization
	{
		const int stacks = 32;
		const int slices = 32;
		const float radius = 1.0f;

		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;

		vertices.reserve((stacks + 1) * (slices + 1));
		indices.reserve(stacks * slices * 6);

		for (int i = 0; i <= stacks; ++i) {
			float phi = M_PI * float(i) / float(stacks); // 0..PI
			float y = cosf(phi);
			float r_sin = sinf(phi);
			for (int j = 0; j <= slices; ++j) {
				float theta = 2.0f * M_PI * float(j) / float(slices); // 0..2PI
				float x = r_sin * cosf(theta);
				float z = r_sin * sinf(theta);
				Vertex v;
				v.position = {radius * x, radius * y, radius * z};
				// normal = normalized position for sphere
				veekay::vec3 n = v.position;
				float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
				if (len > 1e-6f) { n.x /= len; n.y /= len; n.z /= len; }
				v.normal = n;
				// UV spherical coords
				float u = float(j) / float(slices);
				float vcoord = float(i) / float(stacks);
				v.uv = {u, vcoord};
				// Vertical gradient
				float t = (y + 1.0f) * 0.5f; // y in [-1,1] -> t in [0,1]
				v.color = { t, 0.5f * (1.0f - t), 1.0f - t }; 
				vertices.push_back(v);
			}
		}

		for (int i = 0; i < stacks; ++i) {
			for (int j = 0; j < slices; ++j) {
				uint32_t a = uint32_t(i * (slices + 1) + j);
				uint32_t b = uint32_t((i + 1) * (slices + 1) + j);
				uint32_t c = uint32_t((i + 1) * (slices + 1) + (j + 1));
				uint32_t d = uint32_t(i * (slices + 1) + (j + 1));

				indices.push_back(a); indices.push_back(b); indices.push_back(c);
				indices.push_back(a); indices.push_back(c); indices.push_back(d);
			}
		}

		sphere_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		sphere_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		sphere_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Add models to scene
	models.emplace_back(Model{
		.mesh = plane_mesh,
		.transform = Transform{},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.material_id = 0, // obsidian
		.emissive_strength = 0.0f,
		.opacity = 1.0f,
		.specular_scale = 1.0f
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {-2.0f, -0.5f, -1.5f},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.material_id = 0, // obsidian
		.emissive_strength = 0.0f,
		.opacity = 1.0f,
		.specular_scale = 1.0f
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {1.5f, -0.5f, -0.5f},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.material_id = 1, // lava
		.emissive_strength = 1.0f,
		.opacity = 1.0f,
		.specular_scale = -1.0f
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {0.0f, -0.5f, 1.0f},
		},
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
		.material_id = 2, // water
		.emissive_strength = 0.0f,
		.opacity = 0.55f,    // semi transparent
		.specular_scale = 1.6f // stronger highlight
	});

	models.emplace_back(Model{
		.mesh = sphere_mesh,
		.transform = Transform{
			.position = {sphere_position.x, sphere_position.y, sphere_position.z},
			.scale = {1.0f, 1.0f, 1.0f},
		},
		.albedo_color = sphere_color,
		.material_id = 0, // obsidian
		.emissive_strength = 0.0f,
		.opacity = 1.0f,
		.specular_scale = 1.0f
	});

	// default lights
	{
		char* mem = static_cast<char*>(point_lights_ssbo->mapped_region);
		int* header = reinterpret_cast<int*>(mem);
		header[0] = num_point_lights;
		header[1] = 0; header[2] = 0; header[3] = 0;
		PointLightCPU *pls = reinterpret_cast<PointLightCPU*>(mem + 16);

		// One white point light above scene
		PointLightCPU pl{};
		pl.pos_int = veekay::vec4{0.0f, -1.0f, 3.0f, 3.6f};
		pl.color_radius = veekay::vec4{0.26f, 0.36f, 0.95f, 10.0f};
		pl.atten = veekay::vec4{1.0f, 0.09f, 0.032f, 1.0f};
		pls[0] = pl;

		for (uint32_t i = 1; i < 8; ++i) pls[i] = PointLightCPU{};
	}

	{
		char* mem = static_cast<char*>(spot_lights_ssbo->mapped_region);
		int* header = reinterpret_cast<int*>(mem);
		header[0] = num_spot_lights;
		header[1] = 0; header[2] = 0; header[3] = 0;
		SpotLightCPU *sls = reinterpret_cast<SpotLightCPU*>(mem + 16);

		SpotLightCPU sl{};
		sl.pos_int = veekay::vec4{ -2.0f, -7.0f, -1.0f, 2.0f }; 
		veekay::vec3 dir = { 0.0f, 1.0f, 0.0f };
		float innerDeg = 12.0f, outerDeg = 18.0f;
		sl.dir_inner = veekay::vec4{ dir.x, dir.y, dir.z, cosf(innerDeg * M_PI / 180.0f) };
		sl.color_outer = veekay::vec4{ 1.0f, 0.0f, 0.0f, cosf(outerDeg * M_PI / 180.0f) };
		sl.atten = veekay::vec4{1.0f, 0.09f, 0.032f, 1.0f};
		sls[0] = sl;

		for (uint32_t i = 1; i < 4; ++i) sls[i] = SpotLightCPU{};
	}
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	if (material_sampler) vkDestroySampler(device, material_sampler, nullptr);
	if (missing_texture_sampler) vkDestroySampler(device, missing_texture_sampler, nullptr);
	for (auto &mat : materials) {
		if (mat.albedo && mat.albedo != missing_texture) delete mat.albedo;
		if (mat.specular && mat.specular != empty_specular) delete mat.specular;
		if (mat.emissive && mat.emissive != empty_emissive) delete mat.emissive;
	}
	delete missing_texture;
	if (empty_specular_sampler) vkDestroySampler(device, empty_specular_sampler, nullptr);
	delete empty_specular;
	if (empty_emissive_sampler) vkDestroySampler(device, empty_emissive_sampler, nullptr);
	delete empty_emissive;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete plane_mesh.index_buffer;
	delete plane_mesh.vertex_buffer;

	delete sphere_mesh.index_buffer;
	delete sphere_mesh.vertex_buffer;

	delete point_lights_ssbo;
	delete spot_lights_ssbo;

	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	if (scene_set_layout) vkDestroyDescriptorSetLayout(device, scene_set_layout, nullptr);
	if (material_set_layout) vkDestroyDescriptorSetLayout(device, material_set_layout, nullptr);
	if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	current_time = static_cast<float>(time);

	ImGui::Begin("Controls:");

	{ // sphere controls
		ImGui::Text("Sphere:");
		ImGui::InputFloat3("Translation", reinterpret_cast<float*>(&sphere_position));
		ImGui::SliderFloat("Rotation", &sphere_rotation, 0.0f, 2.0f * M_PI);
		ImGui::Checkbox("Spin?", &sphere_spin);

		ImGui::Separator();
		ImGui::ColorEdit3("Sphere color", reinterpret_cast<float*>(&sphere_color));

		ImGui::Separator();
		ImGui::Text("Pulsation:");
		ImGui::SliderFloat("Amplitude", &puls_amp, 0.0f, 2.0f);
		ImGui::SliderFloat("Frequency", &puls_freq, 0.0f, 10.0f);
		ImGui::Text("Scale value: %.3f", scale_value);
	}

	ImGui::Separator();

	// Camera controls
	ImGui::Text("Camera:");
	ImGui::InputFloat3("Cam position", reinterpret_cast<float*>(&camera.position));
	ImGui::SliderFloat("FOV", &camera.fov, 10.0f, 120.0f);
	const char* modes[] = {"LookAt", "TransformInverse"};
	int selected = (camera_view_mode == CameraViewMode::LookAt) ? 0 : 1;
	if (ImGui::Combo("Camera mode", &selected, modes, IM_ARRAYSIZE(modes))) {
		camera_view_mode = (selected == 0) ? CameraViewMode::LookAt : CameraViewMode::TransformInverse;
	}

	ImGui::Separator();
	ImGui::Checkbox("Animate path (figure inf sign)", &animate_path);
	ImGui::SliderFloat("Path amp X", &path_amp_x, 0.0f, 10.0f);
	ImGui::SliderFloat("Path amp Y", &path_amp_y, 0.0f, 5.0f);
	ImGui::SliderFloat("Path speed", &path_speed, 0.0f, 10.0f);

	ImGui::Separator();
	// --- Directional light (простая запись в scene_uniforms позже) ---
	static float dir_dir[3] = {0.3f, -1.0f, -0.4f};
	static bool dir_enabled = true;
	static float dir_color[3] = {1.0f, 1.0f, 1.0f};
	static float dir_intensity = 1.5f;
	ImGui::InputFloat3("Dir light dir", dir_dir);
	ImGui::ColorEdit3("Dir light color", dir_color);
	ImGui::SliderFloat("Dir intensity", &dir_intensity, 0.0f, 5.0f);
	ImGui::Checkbox("Dir enabled", &dir_enabled);


	// --- Point lights: редактируем в-place в SSBO mapped_region ---
	{
		char* mem = static_cast<char*>(point_lights_ssbo->mapped_region);
		int* header = reinterpret_cast<int*>(mem);
		// header[0] — число источников
		ImGui::SliderInt("Num Point Lights", &header[0], 0, 8);

		PointLightCPU* pls = reinterpret_cast<PointLightCPU*>(mem + 16);

		int n = header[0];
		for (int i = 0; i < n; ++i) {
			ImGui::PushID(i);
			ImGui::Text("Point %d", i);
			ImGui::InputFloat3("pos", &pls[i].pos_int.x);               // pos.x,y,z
			ImGui::SliderFloat("intensity", &pls[i].pos_int.w, 0.0f, 10.0f);
			ImGui::ColorEdit3("color", &pls[i].color_radius.x);        // color.xyz
			ImGui::InputFloat("radius", &pls[i].color_radius.w);       // radius in color_radius.w
			ImGui::InputFloat3("atten (c,l,q)", &pls[i].atten.x);      // atten.x,y,z
			bool enabled = pls[i].atten.w > 0.5f;
			if (ImGui::Checkbox("enabled", &enabled)) pls[i].atten.w = enabled ? 1.0f : 0.0f;
			ImGui::Separator();
			ImGui::PopID();
		}
	}

	// --- Spot lights: тоже редактируем прямо в mapped_region ---
	{
		char* mem = static_cast<char*>(spot_lights_ssbo->mapped_region);
		int* header = reinterpret_cast<int*>(mem);
		ImGui::SliderInt("Num Spot Lights", &header[0], 0, 4);

		SpotLightCPU* sls = reinterpret_cast<SpotLightCPU*>(mem + 16);
		int n = header[0];
		for (int i = 0; i < n; ++i) {
			ImGui::PushID(100+i);
			ImGui::Text("Spot %d", i);
			ImGui::InputFloat3("pos", &sls[i].pos_int.x);
			ImGui::SliderFloat("intensity", &sls[i].pos_int.w, 0.0f, 10.0f);
			ImGui::InputFloat3("dir", &sls[i].dir_inner.x); // normalized preferred
			// Уголы храним в косинусах в dir_inner.w и color_outer.w — но в UI удобнее градусы:
			float innerDeg = acosf(sls[i].dir_inner.w) * 180.0f / M_PI;
			float outerDeg = acosf(sls[i].color_outer.w) * 180.0f / M_PI;
			if (ImGui::SliderFloat("inner deg", &innerDeg, 0.0f, 90.0f)) {
				sls[i].dir_inner.w = cosf(innerDeg * M_PI / 180.0f);
			}
			if (ImGui::SliderFloat("outer deg", &outerDeg, 0.0f, 90.0f)) {
				sls[i].color_outer.w = cosf(outerDeg * M_PI / 180.0f);
			}
			ImGui::ColorEdit3("color", &sls[i].color_outer.x);
			ImGui::InputFloat3("atten (c,l,q)", &sls[i].atten.x);
			bool enabled = sls[i].atten.w > 0.5f;
			if (ImGui::Checkbox("enabled", &enabled)) sls[i].atten.w = enabled ? 1.0f : 0.0f;
			ImGui::Separator();
			ImGui::PopID();
		}
	}

	ImGui::End();

	if (!ImGui::IsWindowHovered()) {
		using namespace veekay::input;

		float cp = cosf(camera.rotation.x);
		veekay::vec3 forward = { -cp * sinf(camera.rotation.y), sinf(camera.rotation.x), -cp * cosf(camera.rotation.y) };
		veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(forward, camera.up));
		veekay::vec3 up = veekay::vec3::normalized(veekay::vec3::cross(forward, right));

		if (keyboard::isKeyDown(keyboard::Key::w)) camera.position -= forward * move_speed;
		if (keyboard::isKeyDown(keyboard::Key::s)) camera.position += forward * move_speed;
		if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * move_speed;
		if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * move_speed;
		if (keyboard::isKeyDown(keyboard::Key::q)) camera.position += up * move_speed;
		if (keyboard::isKeyDown(keyboard::Key::z)) camera.position -= up * move_speed;

		if (mouse::isButtonDown(mouse::Button::left)) {
			auto md = mouse::cursorDelta();
			camera.rotation.y += float(md.x) * look_sensitivity;
			camera.rotation.x += float(-md.y) * look_sensitivity;
			const float limit = 1.49f;
			if (camera.rotation.x > limit) camera.rotation.x = limit;
			if (camera.rotation.x < -limit) camera.rotation.x = -limit;
    	}

	}

	// Sphere animation updates
	if (sphere_spin) {
		sphere_rotation = float(time);
	}
	sphere_rotation = fmodf(sphere_rotation, 2.0f * M_PI);

	// Pulsation effect
	scale_value = 1.0f + puls_amp * sinf(current_time * puls_freq);
	if (scale_value <= 0.001f) scale_value = 0.001f;

	// Path animation (figure-eight) - updates sphere_position
	if (animate_path && !prev_animate_path) {
		path_origin = sphere_position;
		path_origin_initialized = true;
	}
	prev_animate_path = animate_path;
	if (animate_path) {
		float t = current_time * path_speed;
		float dx = path_amp_x * cosf(t);
    	float dy = path_amp_y * sinf(t) * cosf(t);

		sphere_position.x = path_origin.x + dx;
		sphere_position.y = path_origin.y + dy;
		sphere_position.z = path_origin.z;
	}

	float aspect = float(veekay::app.window_width) / float(veekay::app.window_height);
	SceneUniforms scene_uniforms{};
	scene_uniforms.view_projection = camera.view_projection(aspect);
	// scene_uniforms.camera_position = veekay::vec4{camera.position.x, camera.position.y, camera.position.z, 0.0f};
	scene_uniforms.camera_position = veekay::vec4{camera.position.x, camera.position.y, camera.position.z, 1.0f};

	// simple directional light: from above and slightly front
	scene_uniforms.dir_light_dir = veekay::vec4{ veekay::vec3::normalized({dir_dir[0], dir_dir[1], dir_dir[2]}).x,
                                            veekay::vec3::normalized({dir_dir[0], dir_dir[1], dir_dir[2]}).y,
                                            veekay::vec3::normalized({dir_dir[0], dir_dir[1], dir_dir[2]}).z,
                                            dir_enabled ? 1.0f : 0.0f };
	scene_uniforms.dir_light_color = veekay::vec4{ dir_color[0], dir_color[1], dir_color[2], dir_intensity };
	scene_uniforms.time = current_time;


	// Update sphere model transform in models array (sphere placed at last index)
	if (!models.empty()) {
		size_t sphere_index = models.size() - 1;
		Model& sphere_model = models[sphere_index];
		sphere_model.transform.position = sphere_position;
		sphere_model.transform.scale = veekay::vec3{scale_value, scale_value, scale_value};
		sphere_model.transform.rotation.y = sphere_rotation;
		// update albedo color from UI
		sphere_model.albedo_color = sphere_color;
	}

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model &m = models[i];
		ModelUniforms u{};
		u.model = m.transform.matrix();
		u.albedo_color = veekay::vec4{m.albedo_color.x, m.albedo_color.y, m.albedo_color.z, 0.0f};
		u.specular_color = veekay::vec4{m.specular_color.x, m.specular_color.y, m.specular_color.z, 0.0f};
		// misc.x = shininess; misc.y = emissive_strength; misc.z = opacity; misc.w = specular_scale
		u.misc = veekay::vec4{ m.shininess, m.emissive_strength, m.opacity, m.specular_scale };
		model_uniforms[i] = u;
	}

	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

	const size_t alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
		const ModelUniforms& uniforms = model_uniforms[i];

		char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
		*reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
	}
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * model_uniorms_alignment;
		// bind scene set (set = 0) with dynamic offset
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &scene_descriptor_set, 1, &offset);
		
		// bind material set (set = 1) for this model
		int matId = model.material_id;
		if (matId < 0 || matId >= (int)materials.size()) matId = 0;
		VkDescriptorSet matSet = materials[matId].descriptor_set;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        1, 1, &matSet, 0, nullptr);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
