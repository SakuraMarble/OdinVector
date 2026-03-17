set(CMAKE_CXX_STANDARD 17)

add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ${PROJECT_NAME})

add_executable(search_memory_index search_memory_index.cpp)
target_link_libraries(search_memory_index ${PROJECT_NAME})

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME})

add_executable(gen_tags gen_tags.cpp)
target_link_libraries(gen_tags ${PROJECT_NAME})

add_executable(gt_update gt_update.cpp)
target_link_libraries(gt_update ${PROJECT_NAME} )

add_executable(convert_bin convert_bin.cpp)
target_link_libraries(convert_bin ${PROJECT_NAME} )

add_executable(change_pts change_pts.cpp)
target_link_libraries(change_pts ${PROJECT_NAME})

add_executable(overall_performance overall_performance.cpp)
target_link_libraries(overall_performance ${PROJECT_NAME})

add_executable(test_insert_search test_insert_search.cpp)
target_link_libraries(test_insert_search ${PROJECT_NAME})

add_executable(overall_perf_mem overall_perf_mem.cpp)
target_link_libraries(overall_perf_mem ${PROJECT_NAME})

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME})

add_executable(search_disk_index_mem search_disk_index_mem.cpp)
target_link_libraries(search_disk_index_mem ${PROJECT_NAME})

add_executable(pad_partition pad_partition.cpp)
target_link_libraries(pad_partition ${PROJECT_NAME} )

# CRoaring-based hybrid tag-vector search
set(CROARING_ROOT "/home/mabojing/.local/lib/croaring" CACHE PATH "CRoaring installation root")
find_library(CROARING_LIB
    NAMES roaring
    PATHS ${CROARING_ROOT}/lib
    NO_DEFAULT_PATH)
find_path(CROARING_INCLUDE
    NAMES roaring/roaring.hh
    PATHS ${CROARING_ROOT}/include
    NO_DEFAULT_PATH)

if(NOT CROARING_LIB OR NOT CROARING_INCLUDE)
    message(WARNING "CRoaring not found; hybrid_tag_search and build_ivf_index will not be built.")
else()
    message(STATUS "CRoaring found: lib=${CROARING_LIB}, include=${CROARING_INCLUDE}")
    
    # 1. 编译 hybrid_tag_search
    add_executable(hybrid_tag_search
        hybrid_tag_search.cpp
        InvertedIndex.cpp
    )
    target_include_directories(hybrid_tag_search PRIVATE ${CROARING_INCLUDE})
    target_link_libraries(hybrid_tag_search ${PROJECT_NAME} ${CROARING_LIB})

    # 2. 编译 build_ivf_index (注意这里新增了 InvertedIndex.cpp)
    add_executable(build_ivf_index 
        build_ivf_index.cpp 
        InvertedIndex.cpp
    )
    target_include_directories(build_ivf_index PRIVATE ${CROARING_INCLUDE})
    target_link_libraries(build_ivf_index ${PROJECT_NAME} ${CROARING_LIB})

    add_executable(test_dynamic_ivf
        test_dynamic_ivf.cpp
        InvertedIndex.cpp
    )
    target_include_directories(test_dynamic_ivf PRIVATE ${CROARING_INCLUDE})
    target_link_libraries(test_dynamic_ivf ${PROJECT_NAME} ${CROARING_LIB})
endif()