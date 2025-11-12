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

// struct Matrix {
// 	float m[4][4];
// };

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
};

// std140 ModelUniforms (binding = 1 in shader)
struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec4 albedo_color;
	veekay::vec4 specular_color;
	veekay::vec4 misc;
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

// struct SceneUniforms {
// 	veekay::mat4 view_projection;
// };

// struct ModelUniforms {
// 	veekay::mat4 model;
// 	veekay::vec3 albedo_color; float _pad0;
// };

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

// Lighting defaults
int num_point_lights = 1;
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
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

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
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

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

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
	// TODO: Scaling and rotation

	auto t = veekay::mat4::translation(position);
	auto s = veekay::mat4::scaling(scale);
    veekay::mat4 rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
    veekay::mat4 ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
    veekay::mat4 rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);
	veekay::mat4 r = rz * ry * rx;

	return s * r * t;
}

veekay::mat4 Camera::view() const {
	// TODO: Rotation

	// auto t = veekay::mat4::translation(-position);
	auto t = veekay::mat4::translation(position);
	auto rx = veekay::mat4::rotation({1,0,0}, rotation.x);
	auto ry = veekay::mat4::rotation({0,1,0}, rotation.y);
	auto rz = veekay::mat4::rotation({0,0,1}, rotation.z);
	auto cam = t * rz * ry * rx;
	return veekay::mat4::inverse(cam);
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * projection;
}

// Matrix transpose(const Matrix &a) {
//     Matrix r{};
//     for (int i = 0; i < 4; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             r.m[i][j] = a.m[j][i];
//         }
//     }
//     return r;
// }

veekay::vec3 cross(const veekay::vec3& a, const veekay::vec3& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

veekay::vec3 normalize(const veekay::vec3& v) {
	float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	return { v.x / length, v.y / length, v.z / length };
}

veekay::vec3 add(const veekay::vec3& a, const veekay::vec3& b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

float dot(const veekay::vec3& a, const veekay::vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Matrix lookAt(const Vector& eye, const Vector& center, const Vector& up) {
//     Vector f = normalize({ center.x - eye.x, center.y - eye.y, center.z - eye.z }); // forward
//     Vector s = normalize(cross(f, up)); // right
//     Vector u = cross(s, f);             // true up

//     Matrix m = identity();

//     // Заполняем матрицу в row-vector формате:
//     // первые 3 строки — базис (right, up, -forward) как строки,
//     // последняя строка — перевод (dot с отрицанием/положением).
//     m.m[0][0] = s.x; m.m[0][1] = s.y; m.m[0][2] = s.z; m.m[0][3] = 0.0f;
//     m.m[1][0] = u.x; m.m[1][1] = u.y; m.m[1][2] = u.z; m.m[1][3] = 0.0f;
//     m.m[2][0] = -f.x; m.m[2][1] = -f.y; m.m[2][2] = -f.z; m.m[2][3] = 0.0f;

//     // translation row (last row) — преобразование точки в систему камеры
//     m.m[3][0] = -dot(eye, s);
//     m.m[3][1] = -dot(eye, u);
//     m.m[3][2] =  dot(eye, f);
//     m.m[3][3] = 1.0f;

//     return m;
// }

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
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 8,
				}
			};
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 4,
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

		// NOTE: Descriptor set layout specification
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
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
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
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
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
		.albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f}
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {-2.0f, -0.5f, -1.5f},
		},
		.albedo_color = veekay::vec3{1.0f, 0.0f, 0.0f}
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {1.5f, -0.5f, -0.5f},
		},
		.albedo_color = veekay::vec3{0.0f, 1.0f, 0.0f}
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {0.0f, -0.5f, 1.0f},
		},
		.albedo_color = veekay::vec3{0.0f, 0.0f, 1.0f}
	});

	models.emplace_back(Model{
		.mesh = sphere_mesh,
		.transform = Transform{
			.position = {sphere_position.x, sphere_position.y, sphere_position.z},
			.scale = {1.0f, 1.0f, 1.0f},
		},
		.albedo_color = sphere_color
	});

	{
		char* mem = static_cast<char*>(point_lights_ssbo->mapped_region);
		int* header = reinterpret_cast<int*>(mem);
		header[0] = num_point_lights;
		header[1] = 0; header[2] = 0; header[3] = 0;
		PointLightCPU *pls = reinterpret_cast<PointLightCPU*>(mem + 16);

		// One white point light above scene
		PointLightCPU pl{};
		pl.pos_int = veekay::vec4{0.0f, 3.0f, 0.0f, 2.0f};
		pl.color_radius = veekay::vec4{1.0f, 1.0f, 1.0f, 10.0f};
		pl.atten = veekay::vec4{1.0f, 0.09f, 0.032f, 1.0f};
		pls[0] = pl;

		for (uint32_t i = 1; i < 8; ++i) pls[i] = PointLightCPU{};
	}

	{
		char* mem = static_cast<char*>(spot_lights_ssbo->mapped_region);
		int* header = reinterpret_cast<int*>(mem);
		header[0] = num_spot_lights;
		header[1] = 0; header[2] = 0; header[3] = 0;
		// No spot lights by default
		SpotLightCPU *sls = reinterpret_cast<SpotLightCPU*>(mem + 16);
		for (uint32_t i = 0; i < 4; ++i) sls[i] = SpotLightCPU{};
	}
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

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

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

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

	ImGui::Separator();
	ImGui::Checkbox("Animate path (figure inf sign)", &animate_path);
	ImGui::SliderFloat("Path amp X", &path_amp_x, 0.0f, 10.0f);
	ImGui::SliderFloat("Path amp Y", &path_amp_y, 0.0f, 5.0f);
	ImGui::SliderFloat("Path speed", &path_speed, 0.0f, 10.0f);
	ImGui::End();

	if (!ImGui::IsWindowHovered()) {
		using namespace veekay::input;

		if (mouse::isButtonDown(mouse::Button::left)) {
			auto move_delta = mouse::cursorDelta();

			// TODO: Use mouse_delta to update camera rotation
			
			auto view = camera.view();

		}
		// TODO: Calculate right, up and front from view matrix
		veekay::vec3 right = {1.0f, 0.0f, 0.0f};
		veekay::vec3 up = {0.0f, -1.0f, 0.0f};
		veekay::vec3 front = {0.0f, 0.0f, 1.0f};

		if (keyboard::isKeyDown(keyboard::Key::w))
			camera.position += front * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::s))
			camera.position -= front * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::d))
			camera.position += right * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::a))
			camera.position -= right * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::q))
			camera.position += up * 0.1f;

		if (keyboard::isKeyDown(keyboard::Key::z))
			camera.position -= up * 0.1f;
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
	scene_uniforms.camera_position = veekay::vec4{camera.position.x, camera.position.y, camera.position.z, 0.0f};

	// simple directional light: from above and slightly front
	scene_uniforms.dir_light_dir = veekay::vec4{ normalize(veekay::vec3{0.3f, -1.0f, -0.4f}).x,
	                                   normalize(veekay::vec3{0.3f, -1.0f, -0.4f}).y,
	                                   normalize(veekay::vec3{0.3f, -1.0f, -0.4f}).z,
	                                   0.0f };
	scene_uniforms.dir_light_color = veekay::vec4{1.0f, 1.0f, 0.9f, 0.6f};

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
		u.misc = veekay::vec4{m.shininess, 0.0f, 0.0f, 0.0f};
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

	{
		char* mem = static_cast<char*>(point_lights_ssbo->mapped_region);
		int* header = reinterpret_cast<int*>(mem);
		header[0] = num_point_lights;
		header[1] = header[2] = header[3] = 0;
		PointLightCPU* pls = reinterpret_cast<PointLightCPU*>(mem + 16);
		PointLightCPU pl{};
		pl.pos_int = veekay::vec4{ 0.0f, 3.0f, 0.0f, 2.0f };
		pl.color_radius = veekay::vec4{ 1.0f, 1.0f, 0.8f, 10.0f };
		pl.atten = veekay::vec4{ 1.0f, 0.09f, 0.032f, 1.0f };
		pls[0] = pl;
		for (uint32_t i = 1; i < 8; ++i) pls[i] = PointLightCPU{};
	}

	{
		char* mem = static_cast<char*>(spot_lights_ssbo->mapped_region);
		int* header = reinterpret_cast<int*>(mem);
		header[0] = num_spot_lights;
		header[1] = header[2] = header[3] = 0;
		SpotLightCPU* sls = reinterpret_cast<SpotLightCPU*>(mem + 16);
		for (uint32_t i = 0; i < 4; ++i) sls[i] = SpotLightCPU{};
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
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &descriptor_set, 1, &offset);

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
