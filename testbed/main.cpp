#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

constexpr float M_PI = 3.14159265358979323846f;

namespace {

constexpr float camera_fov = 70.0f;
constexpr float camera_near_plane = 0.01f;
constexpr float camera_far_plane = 100.0f;

struct Matrix {
	float m[4][4];
};

struct Vector {
	float x, y, z;
};

struct Vertex {
	Vector position;
	// NOTE: You can add more attributes
};

// NOTE: These variable will be available to shaders through push constant uniform
// struct ShaderConstants {
// 	Matrix projection;
// 	Matrix transform;
// 	Vector color;
// };

// NOTE: 16 bytes alignment for vec3
struct ShaderConstants {
	Matrix mvp;
	alignas(16) Vector color;
};

struct VulkanBuffer {
	VkBuffer buffer;
	VkDeviceMemory memory;
};

VkShaderModule vertex_shader_module;
VkShaderModule fragment_shader_module;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

// NOTE: Declare buffers and other variables here
VulkanBuffer vertex_buffer;
VulkanBuffer index_buffer;
uint32_t index_count = 0u;

Vector model_position = {0.0f, 0.0f, 0.0f};
float model_rotation;
Vector model_color = {255.0f, 255.0f, 255.0f };
bool model_spin = true;
float puls_amp = 0.3f;
float puls_freq = 1.5f;
float scale_value = 1.0f;

float camera_yaw = 0.0f;
float camera_pitch = 0.0f;
float camera_distance = 5.0f;

float current_time = 0.0f;

Matrix identity() {
	Matrix result{};

	result.m[0][0] = 1.0f;
	result.m[1][1] = 1.0f;
	result.m[2][2] = 1.0f;
	result.m[3][3] = 1.0f;
	
	return result;
}

Matrix projection(float fov, float aspect_ratio, float near, float far) {
	Matrix result{};

	const float radians = fov * M_PI / 180.0f;
	const float cot = 1.0f / tanf(radians / 2.0f);

	result.m[0][0] = cot / aspect_ratio;
	result.m[1][1] = cot;
	result.m[2][3] = 1.0f;

	result.m[2][2] = far / (far - near);
	result.m[3][2] = (-near * far) / (far - near);

	return result;
}

Matrix translation(Vector vector) {
	Matrix result = identity();

	result.m[3][0] = vector.x;
	result.m[3][1] = vector.y;
	result.m[3][2] = vector.z;

	return result;
}

Matrix scaling(Vector vector) {
	Matrix result = identity();

	result.m[0][0] = vector.x;
	result.m[1][1] = vector.y;
	result.m[2][2] = vector.z;

	return result;
}

Matrix rotation(Vector axis, float angle) {
	Matrix result{};

	float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

	axis.x /= length;
	axis.y /= length;
	axis.z /= length;

	float sina = sinf(angle);
	float cosa = cosf(angle);
	float cosv = 1.0f - cosa;

	result.m[0][0] = (axis.x * axis.x * cosv) + cosa;
	result.m[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
	result.m[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

	result.m[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
	result.m[1][1] = (axis.y * axis.y * cosv) + cosa;
	result.m[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

	result.m[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
	result.m[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
	result.m[2][2] = (axis.z * axis.z * cosv) + cosa;

	result.m[3][3] = 1.0f;

	return result;
}

Matrix multiply(const Matrix& a, const Matrix& b) {
	Matrix result{};

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			float sum = 0.0f;
			for (int k = 0; k < 4; k++) {
				sum += a.m[j][k] * b.m[k][i];
			}
			result.m[j][i] = sum;
		}
	}

	return result;
}

Matrix transpose(const Matrix &a) {
    Matrix r{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            r.m[i][j] = a.m[j][i];
        }
    }
    return r;
}

Vector cross(const Vector& a, const Vector& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

Vector normalize(const Vector& v) {
	float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	return { v.x / length, v.y / length, v.z / length };
}

Vector add(const Vector& a, const Vector& b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

float dot(const Vector& a, const Vector& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

Matrix lookAt(const Vector& eye, const Vector& center, const Vector& up) {
    Vector f = normalize({ center.x - eye.x, center.y - eye.y, center.z - eye.z }); // forward
    Vector s = normalize(cross(f, up)); // right
    Vector u = cross(s, f);             // true up

    Matrix m = identity();

    // Заполняем матрицу в row-vector формате:
    // первые 3 строки — базис (right, up, -forward) как строки,
    // последняя строка — перевод (dot с отрицанием/положением).
    m.m[0][0] = s.x; m.m[0][1] = s.y; m.m[0][2] = s.z; m.m[0][3] = 0.0f;
    m.m[1][0] = u.x; m.m[1][1] = u.y; m.m[1][2] = u.z; m.m[1][3] = 0.0f;
    m.m[2][0] = -f.x; m.m[2][1] = -f.y; m.m[2][2] = -f.z; m.m[2][3] = 0.0f;

    // translation row (last row) — преобразование точки в систему камеры
    m.m[3][0] = -dot(eye, s);
    m.m[3][1] = -dot(eye, u);
    m.m[3][2] =  dot(eye, f);
    m.m[3][3] = 1.0f;

    return m;
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

VulkanBuffer createBuffer(size_t size, void *data, VkBufferUsageFlags usage) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
	
	VulkanBuffer result{};

	{
		// NOTE: We create a buffer of specific usage with specified size
		VkBufferCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};

		if (vkCreateBuffer(device, &info, nullptr, &result.buffer) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan buffer\n";
			return {};
		}
	}

	// NOTE: Creating a buffer does not allocate memory,
	//       only a buffer **object** was created.
	//       So, we allocate memory for the buffer

	{
		// NOTE: Ask buffer about its memory requirements
		VkMemoryRequirements requirements;
		vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

		// NOTE: Ask GPU about types of memory it supports
		VkPhysicalDeviceMemoryProperties properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

		// NOTE: We want type of memory which is visible to both CPU and GPU
		// NOTE: HOST is CPU, DEVICE is GPU; we are interested in "CPU" visible memory
		// NOTE: COHERENT means that CPU cache will be invalidated upon mapping memory region
		const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		// NOTE: Linear search through types of memory until
		//       one type matches the requirements, thats the index of memory type
		uint32_t index = UINT_MAX;
		for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
			const VkMemoryType& type = properties.memoryTypes[i];

			if ((requirements.memoryTypeBits & (1 << i)) &&
			    (type.propertyFlags & flags) == flags) {
				index = i;
				break;
			}
		}

		if (index == UINT_MAX) {
			std::cerr << "Failed to find required memory type to allocate Vulkan buffer\n";
			return {};
		}

		// NOTE: Allocate required memory amount in appropriate memory type
		VkMemoryAllocateInfo info{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = requirements.size,
			.memoryTypeIndex = index,
		};

		if (vkAllocateMemory(device, &info, nullptr, &result.memory) != VK_SUCCESS) {
			std::cerr << "Failed to allocate Vulkan buffer memory\n";
			return {};
		}

		// NOTE: Link allocated memory with a buffer
		if (vkBindBufferMemory(device, result.buffer, result.memory, 0) != VK_SUCCESS) {
			std::cerr << "Failed to bind Vulkan  buffer memory\n";
			return {};
		}

		// NOTE: Get pointer to allocated memory
		void* device_data;
		vkMapMemory(device, result.memory, 0, requirements.size, 0, &device_data);

		memcpy(device_data, data, size);

		vkUnmapMemory(device, result.memory);
	}

	return result;
}

void destroyBuffer(const VulkanBuffer& buffer) {
	VkDevice& device = veekay::app.vk_device;

	vkFreeMemory(device, buffer.memory, nullptr);
	vkDestroyBuffer(device, buffer.buffer, nullptr);
}

void initialize() {
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
			// NOTE: If you want more attributes per vertex, declare them here
#if 0
			{
				.location = 1, // NOTE: Second attribute
				.binding = 0,
				.format = VK_FORMAT_XXX,
				.offset = offset(Vertex, your_attribute),
			},
#endif
		};

		// NOTE: Bring 
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

		// NOTE: Declare constant memory region visible to vertex and fragment shaders
		VkPushConstantRange push_constants{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
			              VK_SHADER_STAGE_FRAGMENT_BIT,
			.size = sizeof(ShaderConstants),
		};

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_constants,
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

	{ // NOTE: Geometry setup
		// TODO: You define model vertices and create buffers here
		// TODO: Index buffer has to be created here too
		// NOTE: Look for createBuffer function

		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		// Vertex vertices[] = {
		// 	{{-1.0f, -1.0f, 0.0f}},
		// 	{{1.0f, -1.0f, 0.0f}},
		// 	{{1.0f, 1.0f, 0.0f}},
		// 	{{-1.0f, 1.0f, 0.0f}},
		// };

		// uint32_t indices[] = { 0, 1, 2, 2, 3, 0 };

		// Parametric sphere
		const int stacks = 9;
		const int slices = 10;
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

		vertex_buffer = createBuffer(vertices.size() * sizeof(Vertex), vertices.data(),
									VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		index_buffer = createBuffer(indices.size() * sizeof(uint32_t), indices.data(),
									VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		
		index_count = static_cast<uint32_t>(indices.size());
	}
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	// NOTE: Destroy resources here, do not cause leaks in your program!
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	current_time = static_cast<float>(time);

	ImGui::Begin("Controls:");
	ImGui::InputFloat3("Translation", reinterpret_cast<float*>(&model_position));
	ImGui::SliderFloat("Rotation", &model_rotation, 0.0f, 2.0f * M_PI);
	ImGui::Checkbox("Spin?", &model_spin);

	// TODO: Your GUI stuff here
	ImGui::Separator();
	ImGui::Text("Camera:");
    ImGui::SliderFloat("Yaw", &camera_yaw, -M_PI, M_PI);
    ImGui::SliderFloat("Pitch", &camera_pitch, -1.4f, 1.4f);
    ImGui::SliderFloat("Distance", &camera_distance, 0.1f, 50.0f);

	ImGui::Separator();
	ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&model_color));

	ImGui::Separator();
	ImGui::Text("Pulsation:");
	ImGui::SliderFloat("Amplitude", &puls_amp, 0.0f, 2.0f);
	ImGui::SliderFloat("Frequency", &puls_freq, 0.0f, 10.0f);
	ImGui::Text("Scale value: %.3f", scale_value);

	ImGui::End();

	// NOTE: Animation code and other runtime variable updates go here
	if (model_spin) {
		model_rotation = float(time);
	}

	model_rotation = fmodf(model_rotation, 2.0f * M_PI);

	// Pulsation effect
	scale_value = 2.0f + puls_amp * sinf(current_time * puls_freq); // Must be > 0
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

	// TODO: Vulkan rendering code here
	// NOTE: ShaderConstant updates, vkCmdXXX expected to be here
	{
		// NOTE: Use our new shiny graphics pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);


		// NOTE: Use our quad vertex buffer
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);

		// NOTE: Use our quad index buffer
		vkCmdBindIndexBuffer(cmd, index_buffer.buffer, offset, VK_INDEX_TYPE_UINT32);

		Vector camera_position = { 0.0f, 0.0f, camera_distance };
		Vector scene_center = model_position;
		Vector forward = { 
			cosf(camera_pitch) * sinf(camera_yaw),
			sinf(camera_pitch),
			cosf(camera_pitch) * cosf(camera_yaw) 
		}; 
		// View direction
		Vector center = add(camera_position, forward);

		Matrix view = lookAt(camera_position, center, {0.0f, 1.0f, 0.0f});


		Matrix scale = scaling({ scale_value, scale_value, scale_value });
		Matrix rot = rotation({ 0.0f, 1.0f, 0.0f }, model_rotation);
		Matrix trans = translation(model_position);

		Matrix model = multiply(trans, multiply(rot, scale));

		float aspect = float(veekay::app.window_width) / float(veekay::app.window_height);
		Matrix proj = projection(camera_fov, aspect, camera_near_plane, camera_far_plane);

		Matrix transform = multiply(model, view);

		Matrix mvp = multiply(transform, proj);

		// NOTE: Variables like model_XXX were declared globally
		ShaderConstants constants{
			// .projection = proj,

			// .transform = multiply(rotation({0.0f, 1.0f, 0.0f}, model_rotation),
			//                       translation(model_position)),
			.mvp = transpose(mvp),

			.color = model_color,
		};

		// NOTE: Update constant memory with new shader constants
		vkCmdPushConstants(cmd, pipeline_layout,
		                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(ShaderConstants), &constants);

		// NOTE: Draw 6 indices (3 vertices * 2 triangles), 1 group, no offsets
		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
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
