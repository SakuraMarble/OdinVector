set(CMAKE_CXX_STANDARD 17)

# add - bench-updates
add_executable(bench_updates bench_updates.cpp)
target_link_libraries(bench_updates ${PROJECT_NAME})
# add hot index
add_executable(hot_update_console hot_update_console.cpp)
target_link_libraries(hot_update_console ${PROJECT_NAME})
# add hot service
add_executable(hot_update_service hot_update_service.cpp)
target_link_libraries(hot_update_service ${PROJECT_NAME})
# add lazy wal recovery test
add_executable(lazy_wal_recovery_test lazy_wal_recovery_test.cpp)
target_link_libraries(lazy_wal_recovery_test ${PROJECT_NAME})
# add lazy wal hot service
add_executable(lazy_wal_hot_service lazy_wal_hot_service.cpp)
target_link_libraries(lazy_wal_hot_service ${PROJECT_NAME})

add_executable(bench_insert_query_concurrent bench_insert_query_concurrent.cpp)
target_link_libraries(bench_insert_query_concurrent ${PROJECT_NAME})

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

# CRoaring-based hybrid tag-vector search.
# Allow users to optionally set CROARING_ROOT, but also search default system paths.
set(CROARING_ROOT "" CACHE PATH "Optional CRoaring installation root")

if(CROARING_ROOT)
    find_library(CROARING_LIB NAMES roaring PATHS ${CROARING_ROOT}/lib)
    find_path(CROARING_INCLUDE NAMES roaring/roaring.hh PATHS ${CROARING_ROOT}/include)
else()
    find_library(CROARING_LIB NAMES roaring)
    find_path(CROARING_INCLUDE NAMES roaring/roaring.hh)
endif()

if(NOT CROARING_LIB OR NOT CROARING_INCLUDE)
    message(WARNING "CRoaring not found; hybrid_tag_search, build_ivf_index, and test_dynamic_ivf will not be built.")
else()
    message(STATUS "CRoaring found: lib=${CROARING_LIB}, include=${CROARING_INCLUDE}")

    add_executable(hybrid_tag_search
        hybrid_tag_search.cpp
        InvertedIndex.cpp
    )
    target_include_directories(hybrid_tag_search PRIVATE ${CROARING_INCLUDE})
    target_link_libraries(hybrid_tag_search ${PROJECT_NAME} ${CROARING_LIB})

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
