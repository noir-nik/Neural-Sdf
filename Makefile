
CC := g++

BUILD_DIR := build
TARGET := neural-sdf
TARGET := neural-sdf

RENDER_DIR := output

INCLUDES := -Iinclude

CXXFLAGS := -MP -MMD -fopenmp -O2 $(INCLUDES) -DNDEBUG

SRCS := $(wildcard source/*.cpp)
OBJS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(notdir $(SRCS))) 

# DEPFILES := $(OBJS:.o=.d)
# -include $(DEPFILES)

ifeq ($(OS), Windows_NT)
LDFLAGS += -fopenmp -lvulkan-1
else
LDFLAGS += -fopenmp -lvulkan
endif

.PHONY: all build run clean

all: build

build: create_dirs $(TARGET)

create_dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(RENDER_DIR)

$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)"
	@$(CC) $(OBJS) $(LDFLAGS) -o $(TARGET)

$(BUILD_DIR)/%.o: source/%.cpp
	@echo "Compiling $(notdir $<)"
	@$(CC) $(CXXFLAGS) -c $< -o $@

shader:
	glslangValidator -V shaders/neural_sdf.comp -o shaders/neural_sdf.spv

run:
	@$(TARGET)

clean:
	@rm -f $(TARGET) $(wildcard $(BUILD_DIR)/*.o) 
# $(wildcard $(RENDER_DIR)/*) $(DEPFILES)