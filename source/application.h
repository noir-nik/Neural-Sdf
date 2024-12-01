#include <vector>
#include <vulkan/vulkan.h>

#include "settings.h"

#define SHADER_PATH "shaders/neural_sdf.spv"

const int WIDTH = SCREEN_WIDTH;
const int HEIGHT = SCREEN_HEIGHT;
const int WORKGROUP_SIZE = 16;
constexpr int num_push_constants = 4;
#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

class ComputeApplication {
private:
  struct Pixel {
    float r, g, b, a;
  };

  VkInstance instance;
  VkDebugReportCallbackEXT debugReportCallback;
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkPipeline pipeline;
  VkPipelineLayout pipelineLayout;
  VkShaderModule computeShaderModule;
  VkCommandPool commandPool;
  VkCommandBuffer commandBuffer;
  VkDescriptorPool descriptorPool;
  VkDescriptorSet descriptorSet;
  VkDescriptorSetLayout descriptorSetLayout;
  VkBuffer bufferPixels, bufferStaging, bufferWeightsDevice;
  VkDeviceMemory bufferMemoryPixels, bufferMemoryStaging,
      bufferMemoryWeightsDevice;
  std::vector<const char *> enabledLayers;
  VkQueue queue;
  struct MLP {
    std::vector<float> w_b;
    int num_hidden_layers;
    int hidden_layer_size;
  };

public:
  MLP mlp;
  ComputeApplication(){};
  void run();
  void saveRenderedImageFromDeviceMemory(VkDevice a_device,
                                                VkDeviceMemory a_bufferMemory,
                                                size_t a_offset, int a_width,
                                                int a_height);
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
      VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType,
      uint64_t object, size_t location, int32_t messageCode,
      const char *pLayerPrefix, const char *pMessage, void *pUserData);
  void createBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice,
                    VkDeviceSize a_size, VkBufferUsageFlags a_usage,
                    VkMemoryPropertyFlags a_properties, VkBuffer &a_buffer,
                    VkDeviceMemory &a_bufferMemory);
  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size,
                  VkQueue queue);
  void createDescriptorSetLayout(VkDevice a_device,
                                        VkDescriptorSetLayout *a_pDSLayout);
  void createDescriptorSets(VkDevice a_device, VkBuffer a_buffer,
                                   size_t a_bufferSize, VkBuffer a_secondBuffer,
                                   size_t a_secondBufferSize,
                                   const VkDescriptorSetLayout *a_pDSLayout,
                                   VkDescriptorPool *a_pDSPool,
                                   VkDescriptorSet *a_pDS);
  void createComputePipeline(VkDevice a_device,
                                    const VkDescriptorSetLayout &a_dsLayout,
                                    VkShaderModule *a_pShaderModule,
                                    VkPipeline *a_pPipeline,
                                    VkPipelineLayout *a_pPipelineLayout);
  void createCommandBuffer(VkDevice a_device, uint32_t queueFamilyIndex,
                                  VkPipeline a_pipeline,
                                  VkPipelineLayout a_layout,
                                  VkCommandPool *a_pool,
                                  VkCommandBuffer *a_pCmdBuff);
  void recordCommandsTo(VkCommandBuffer a_cmdBuff, VkPipeline a_pipeline,
                        VkPipelineLayout a_layout, const VkDescriptorSet &a_ds);
  void submitCommands(VkCommandBuffer a_cmdBuff, VkQueue a_queue,
                             VkDevice a_device);
  void cleanup();
};
