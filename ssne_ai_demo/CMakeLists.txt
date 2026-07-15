cmake_minimum_required(VERSION 3.0.0)

project(ssne_ai_demo)

# 引入路径配置文件
include(cmake_config/Paths.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
# 核心：禁用ABI变更提示
add_compile_options(-Wno-psabi)

# ==========================================
# 头文件搜索路径设置
# ==========================================
include_directories(${M1_SDK_INC_DIR})
include_directories(${M1_SDK_INC_DIR}/smartsoc)

# 添加我们抽离出来的公共组件头文件目录
include_directories(${CMAKE_SOURCE_DIR}/common/include)

# ==========================================
# 收集源文件
# ==========================================
# 1. 搜集 common 下的所有公共核心源文件
file(GLOB COMMON_SRC "${CMAKE_SOURCE_DIR}/common/src/*.cpp")

# 2. 搜集各个业务模块的源文件
file(GLOB BUSINESS_SRC 
    "${CMAKE_SOURCE_DIR}/modules/face_detection/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/face_detection/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/object_detection/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/object_detection/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/optical_flow_debug/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/optical_flow_debug/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/rps_detection/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/rps_detection/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/speed_detection/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/speed_detection/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/facial_expressions/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/facial_expressions/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/gesture_detection/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/gesture_detection/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/focus_tracking/*.cpp"
    "${CMAKE_SOURCE_DIR}/modules/focus_tracking/src/*.cpp"
)

# Keep the EyeDet/FaceID engine explicit so an existing Buildroot/CMake cache
# notices the newly added source file; remove the duplicate produced by GLOB.
list(APPEND BUSINESS_SRC
    "${CMAKE_SOURCE_DIR}/modules/focus_tracking/src/eyedet_faceid_engine.cpp"
)
list(REMOVE_DUPLICATES BUSINESS_SRC)

# ==========================================
# 生成可执行文件并链接库
# ==========================================
# 将统一入口 main.cpp、公共组件和所有业务代码一起编译
add_executable(ssne_ai_demo 
    ${PROJECT_SOURCE_DIR}/main.cpp 
    ${COMMON_SRC} 
    ${BUSINESS_SRC}
)

# 链接依赖的硬件底层动态库
target_link_libraries(${PROJECT_NAME} 
    ${M1_SSNE_LIB}
    ${M1_CMABUFFER_LIB}
    ${M1_OSD_LIB}
    ${M1_SSZLOG_LIB}
    ${M1_ZLOG_LIB}
    ${M1_EMB_LIB}
)

install(TARGETS ssne_ai_demo DESTINATION bin)
