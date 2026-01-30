#pragma once

#include <vk_types.h>
#include <vk_descriptors.h>
#include <vk_loader.h>
#include <camera.h>

// Forward declarations
class VulkanEngine;

// Constants
constexpr unsigned int FRAME_OVERLAP = 2;

// Utility structures
struct DeletionQueue {
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)();
		}
		deletors.clear();
	}
};

// Compute structures
struct ComputePushConstants {
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct ComputeEffect {
	const char* name;
	VkPipeline pipeline;
	VkPipelineLayout layout;
	ComputePushConstants data;
};

// Scene and rendering structures
struct GPUSceneData {
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection;
	glm::vec4 sunlightColor;
};

struct RenderObject {
	uint32_t indexCount;
	uint32_t firstIndex;
	VkBuffer indexBuffer;
	MaterialInstance* material;
	glm::mat4 transform;
	VkDeviceAddress vertexBufferAddress;
};

struct DrawContext {
	std::vector<RenderObject> OpaqueSurfaces;
	std::vector<RenderObject> TransparentSurfaces;
};

// Frame data structure
struct FrameData {
	VkSemaphore _swapchainSemaphore, _renderSemaphore;
	VkFence _renderFence;
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;
	DeletionQueue _deletionQueue;
	DescriptorAllocatorGrowable _frameDescriptors;
};

// Material system
struct GLTFMetallic_Roughness {
	MaterialPipeline opaquePipeline;
	MaterialPipeline transparentPipeline;
	VkDescriptorSetLayout materialLayout;

	struct MaterialConstants {
		glm::vec4 colorFactors;
		glm::vec4 metal_rough_factors;
		glm::vec4 extra[14]; // padding for uniform buffers
	};

	struct MaterialResources {
		AllocatedImage colorImage;
		VkSampler colorSampler;
		AllocatedImage metalRoughImage;
		VkSampler metalRoughSampler;
		VkBuffer dataBuffer;
		uint32_t dataBufferOffset;
	};

	DescriptorWriter writer;

	void build_pipelines(VulkanEngine* engine);
	void clear_resources(VkDevice device);
	MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
};

// Scene node extension
struct MeshNode : public Node {
	std::shared_ptr<MeshAsset> mesh;
	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

// Main engine class
class VulkanEngine {
public:
	// Core state
	bool _isInitialized{ false };
	int _frameNumber{ 0 };
	bool stop_rendering{ false };
	bool resize_requested{ false };

	// Window and rendering
	VkExtent2D _windowExtent{ 1700, 900 };
	VkExtent2D _drawExtent;
	float renderScale = 1.f;
	struct SDL_Window* _window{ nullptr };

	// Camera
	Camera mainCamera;

	// Frame management
	FrameData _frames[FRAME_OVERLAP];
	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; }

	// Vulkan core objects
	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;
	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	// Swapchain
	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	// Rendering resources
	DrawContext mainDrawContext;
	AllocatedImage _drawImage;
	AllocatedImage _depthImage;

	// Scene management
	std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;
	std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;

	// Materials
	MaterialInstance defaultData;
	GLTFMetallic_Roughness metalRoughMaterial;

	// Memory management
	VmaAllocator _allocator;
	DeletionQueue _mainDeletionQueue;

	// Descriptors
	DescriptorAllocatorGrowable globalDescriptorAllocator;
	VkDescriptorSet _drawImageDescriptors;
	VkDescriptorSetLayout _drawImageDescriptorLayout;
	VkDescriptorSetLayout _singleImageDescriptorLayout;
	VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

	// Scene data
	GPUSceneData sceneData;

	// Pipelines
	VkPipeline _gradientPipeline;
	VkPipelineLayout _gradientPipelineLayout;
	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;

	// Immediate submit
	VkFence _immFence;
	VkCommandBuffer _immCommandBuffer;
	VkCommandPool _immCommandPool;

	// Background effects
	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{ 0 };

	// Default textures
	AllocatedImage _whiteImage;
	AllocatedImage _blackImage;
	AllocatedImage _greyImage;
	AllocatedImage _errorCheckerboardImage;

	// Default samplers
	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;

	// Static instance getter
	static VulkanEngine& Get();

	// Public interface
	void init();
	void cleanup();
	void run();

	// Drawing
	void draw();
	void draw_background(VkCommandBuffer cmd);
	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
	void draw_geometry(VkCommandBuffer cmd);
	void update_scene();

	// Resource management
	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void destroy_buffer(const AllocatedBuffer& buffer);
	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	void destroy_image(const AllocatedImage& img);
	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	// Immediate submit
	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

private:
	// Initialization
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void init_descriptors();
	void init_pipelines(); // TODO: Move pipeline initialisation to here
	void init_imgui();
	void init_mesh_pipeline();
	void init_default_data();

	// Swapchain management
	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
	void resize_swapchain();
};

