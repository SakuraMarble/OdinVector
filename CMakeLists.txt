cmake_minimum_required(VERSION 3.0)
project(pipeann)

set(CMAKE_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_C_COMPILER_LAUNCHER ccache)
# set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

include_directories(
	${PROJECT_SOURCE_DIR}/include 
	${PROJECT_SOURCE_DIR}/include/tsl/include 
)

add_compile_options(
	-fopenmp 
	-fopenmp-simd
	-Wall
	-Wextra
	-Wfatal-errors
	-Wno-class-memaccess
	-Wno-unused-variable
	-Wno-unused-parameter
	-Wno-unused-but-set-variable
)
add_compile_options(-g -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops)

message("Additional definitions: " $ENV{ADDITIONAL_DEFINITIONS})
add_definitions(
	$ENV{ADDITIONAL_DEFINITIONS}
	-DNDEBUG
	# -DUSE_AIO
	# -DIN_PLACE_RECORD_UPDATE # disable design #1
	# -DDIRECT_READ_CC # disable opt #1.1
	# -DBG_IO_THREAD # enable opt #1.2
	# -DDELTA_PRUNING # enable opt #2

	# -DNO_POLLUTE_ORIGINAL

	# optimizations.
	# -DREAD_ONLY_TESTS
	# -DNO_MAPPING # starling: must disable this!
	-DOVERLAP_INIT
	-DDYN_PIPE_WIDTH
	
	# policy.
	# -DSTATIC_POLICY

	# unoptimizations.
	# -DNAIVE_PIPE
)

include_directories(
	${PROJECT_SOURCE_DIR}/third_party/liburing/src/include
)

set(BUNDLED_LIBURING_A "${PROJECT_SOURCE_DIR}/third_party/liburing/src/liburing.a")
set(BUNDLED_LIBURING_SO "${PROJECT_SOURCE_DIR}/third_party/liburing/src/liburing.so")

if(EXISTS "${BUNDLED_LIBURING_A}")
	message(STATUS "Using bundled liburing: ${BUNDLED_LIBURING_A}")
	link_libraries("${BUNDLED_LIBURING_A}")
elseif(EXISTS "${BUNDLED_LIBURING_SO}")
	message(STATUS "Using bundled liburing: ${BUNDLED_LIBURING_SO}")
	link_libraries("${BUNDLED_LIBURING_SO}")
else()
	find_library(SYSTEM_LIBURING uring)
	if(SYSTEM_LIBURING)
		message(STATUS "Using system liburing: ${SYSTEM_LIBURING}")
		link_libraries(${SYSTEM_LIBURING})
	else()
		message(FATAL_ERROR "liburing not found. Build bundled liburing (third_party/liburing) or install system liburing-dev.")
	endif()
endif()

link_libraries(
	aio
	jemalloc
)
# add_compile_options(-fsanitize=address,undefined -lasan -lubsan)
# add_link_options(-fsanitize=address,undefined)
# add_compile_options(-fsanitize=undefined -lubsan)
# add_link_options(-fsanitize=undefined)

# check for AVX512 and AVX2, add the corresponding definitions.
# First, for the compiler.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

# Then, for the runtime.
set(CPU_SUPPORTS_AVX512 FALSE)
set(CPU_SUPPORTS_AVX2 FALSE)
try_run(
	RUN_RESULT
	COMPILE_RESULT
	"${CMAKE_BINARY_DIR}"
	SOURCES "${PROJECT_SOURCE_DIR}/tests/test_cpu.cpp"
	COMPILE_OUTPUT_VARIABLE COMPILE_OUT
	RUN_OUTPUT_VARIABLE RUN_OUT
)

if(COMPILE_RESULT AND RUN_RESULT EQUAL 3) # AVX512 + AVX2
	set(CPU_SUPPORTS_AVX512 TRUE)
	message(STATUS "CPU supports AVX512")
elseif(COMPILE_RESULT AND RUN_RESULT EQUAL 1) # AVX2
	set(CPU_SUPPORTS_AVX2 TRUE)
	message(STATUS "CPU supports AVX2")
else()
	message(STATUS "CPU does not support AVX512 or detection failed")
endif()

if(COMPILER_SUPPORTS_AVX512F AND CPU_SUPPORTS_AVX512)
    message(STATUS "Compiler and CPU supports -mavx512f. Enabling AVX512F.")
	add_definitions(-DUSE_AVX512)
elseif(COMPILER_SUPPORTS_AVX2 AND CPU_SUPPORTS_AVX2)
	message(STATUS "Compiler and CPU supports -mavx2. Enabling AVX2.")
	add_definitions(-DUSE_AVX2)
else()
    message(STATUS "Compiler or CPU do not support -mavx512f and -mavx2.")
endif()

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()

	
find_package(BLAS)
if(BLAS_FOUND)
    message(STATUS "BLAS found")
    link_libraries(${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "BLAS is required but was not found.")
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tests/utils)
# add_subdirectory(python)
