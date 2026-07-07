cmake_minimum_required(VERSION 3.10)
include(../../../../nncase/examples/cmake/Riscv64.cmake)

project(falldown_ai_inference C CXX)

set(nncase_sdk_root "${PROJECT_SOURCE_DIR}/../../../../nncase")
set(k230_sdk        "${nncase_sdk_root}/../../../")
set(cdk_src_root    "${k230_sdk}/src/common/cdk")
set(rtsmart_src_root "${k230_sdk}/src/big/rt-smart")

set(CMAKE_EXE_LINKER_FLAGS
    "-T ${nncase_sdk_root}/examples/cmake/link.lds --static")

# 完整保留原有的src目录结构
set(src
    src/main.cc
    src/ai_inference.cc
)

set(bin falldown_detect.elf)
set(ai_event_bin falldown_ai_event.elf)
set(video_display_bin falldown_video_display.elf)
set(split_launcher_bin falldown_split_launcher.elf)

set(CMAKE_INSTALL_PREFIX
    "${PROJECT_SOURCE_DIR}/../"
    CACHE STRING "The path to use for make install" FORCE)

set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -fopenmp -march=rv64imafdcv -mabi=lp64d -mcmodel=medany"
    CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")

# ====================== 关键修改1：添加C++17标准（NNCASE必须） ======================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ====================== 关键修改2：移除有硬编码bug的find_package ======================
# 移除了OpenCV_DIR和find_package，改用手动头文件路径
include_directories(${nncase_sdk_root}/riscv64)
include_directories(${nncase_sdk_root}/riscv64/nncase/include)
include_directories(${nncase_sdk_root}/riscv64/nncase/include/nncase/runtime)
include_directories(${PROJECT_SOURCE_DIR}/include)  # 完整保留原有的项目include路径
include_directories(${PROJECT_SOURCE_DIR}/third_party/k230_comm/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/k230_comm/server)
include_directories(${nncase_sdk_root}/riscv64/rvvlib/include)

# 手动添加OpenCV头文件路径（替代${OpenCV_INCLUDE_DIRS}）
include_directories(${k230_sdk}/src/big/utils/lib/opencv_thead/include)
include_directories(${k230_sdk}/src/big/utils/lib/opencv_thead/include/opencv4)
include_directories(${k230_sdk}/src/big/utils/lib/opencv/include)
include_directories(${k230_sdk}/src/big/utils/lib/opencv/include/opencv4)

include_directories(${k230_sdk}/src/big/mpp/userapps/api/)
include_directories(${k230_sdk}/src/big/mpp/include)
include_directories(${k230_sdk}/src/big/mpp/include/comm)
include_directories(${k230_sdk}/src/big/mpp/userapps/sample/sample_vo)
include_directories(${cdk_src_root}/user/component/ipcmsg/include)
include_directories(${cdk_src_root}/user/component/datafifo/include)
include_directories(${cdk_src_root}/user/mapi/include)
include_directories(${cdk_src_root}/user/mapi/include/api)

# 完整保留所有原有的库路径（包括hhb-prebuilt-decode）
link_directories(${k230_sdk}/src/big/mpp/userapps/lib)
link_directories(${cdk_src_root}/user/mapi/mediaserver/lib)
link_directories(${cdk_src_root}/user/component/ipcmsg/slave/lib)
link_directories(${cdk_src_root}/user/component/datafifo/host/lib)
link_directories(${rtsmart_src_root}/userapps/sdk/rt-thread/lib/risc-v/rv64)
link_directories(${nncase_sdk_root}/riscv64/rvvlib/)
link_directories(${nncase_sdk_root}/riscv64/nncase/lib/)
link_directories(${nncase_sdk_root}/riscv64/opencv/lib/)
link_directories(${k230_sdk}/src/big/utils/lib/opencv_thead/lib/)
link_directories(${k230_sdk}/src/big/utils/lib/opencv_thead/lib/opencv4/3rdparty/)
link_directories(${k230_sdk}/src/big/utils/lib/opencv/lib/)
link_directories(${k230_sdk}/src/big/utils/lib/hhb-prebuilt-decode/lib/)

add_executable(${bin} ${src})
add_executable(${ai_event_bin}
    src/falldown_ai_event.cc
    src/fdd_ipc.cc
    src/ai_inference.cc
)
add_executable(${video_display_bin}
    src/falldown_video_display.cc
    src/fdd_ipc.cc
    third_party/k230_comm/server/comm_server.cpp
    third_party/k230_comm/server/event_writer.cpp
    third_party/k230_comm/server/avdata_writer.cpp
    third_party/k230_comm/server/avdata_reader.cpp
    third_party/k230_comm/server/data_fifo_writer.cpp
    third_party/k230_comm/server/data_fifo_reader.cpp
)
add_executable(${split_launcher_bin}
    src/falldown_split_launcher.cc
)

# 完整保留所有原有的MPP和NNCASE库链接
target_link_libraries(${bin}
    -Wl,--start-group
    rvv
    Nncase.Runtime.Native
    nncase.rt_modules.k230
    functional_k230
    sys vicap vb cam_device cam_engine
    hal oslayer ebase fpga isp_drv binder auto_ctrol common cam_caldb isi
    3a buffer_management cameric_drv video_in virtual_hal start_engine
    cmd_buffer switch cameric_reg_drv t_database_c t_mxml_c t_json_c
    t_common_c vo connector sensor atomic dma
    -Wl,--end-group
)

target_link_libraries(${ai_event_bin}
    -Wl,--start-group
    rvv
    Nncase.Runtime.Native
    nncase.rt_modules.k230
    functional_k230
    sys vicap vb cam_device cam_engine
    hal oslayer ebase fpga isp_drv binder auto_ctrol common cam_caldb isi
    3a buffer_management cameric_drv video_in virtual_hal start_engine
    cmd_buffer switch cameric_reg_drv t_database_c t_mxml_c t_json_c
    t_common_c vo connector sensor atomic dma
    -Wl,--end-group
)

target_link_libraries(${video_display_bin}
    -Wl,--start-group
    mapi
    datafifo
    ipcmsg
    rvv
    Nncase.Runtime.Native
    nncase.rt_modules.k230
    functional_k230
    sys vicap vb cam_device cam_engine
    hal oslayer ebase fpga isp_drv binder auto_ctrol common cam_caldb isi
    3a buffer_management cameric_drv video_in virtual_hal start_engine
    cmd_buffer switch cameric_reg_drv t_database_c t_mxml_c t_json_c
    t_common_c vo connector sensor atomic dma
    rtthread
    -Wl,--end-group
)

# ====================== 关键修改3：100%匹配你找到的库文件名 ======================
target_link_libraries(${bin}
    # OpenCV核心库
    opencv_videoio
    opencv_imgcodecs
    opencv_imgproc
    opencv_core

    # OpenCV依赖库（完全匹配你找到的文件名）
    csi_cv          # 对应 libcsi_cv.a
    libpng          # 对应 liblibpng.a
    libjpeg-turbo   # 对应 liblibjpeg-turbo.a
    libtiff         # 对应 liblibtiff.a
    libwebp         # 对应 liblibwebp.a
    libopenjp2      # 对应 liblibopenjp2.a
    zlib            # 对应 libzlib.a

    # 系统库（静态链接必须）
    pthread
    dl
    m
    stdc++
)

target_link_libraries(${ai_event_bin}
    # OpenCV核心库
    opencv_videoio
    opencv_imgcodecs
    opencv_imgproc
    opencv_core

    # OpenCV依赖库（完全匹配你找到的文件名）
    csi_cv
    libpng
    libjpeg-turbo
    libtiff
    libwebp
    libopenjp2
    zlib

    # 系统库（静态链接必须）
    pthread
    dl
    m
    stdc++
)

target_link_libraries(${video_display_bin}
    # OpenCV核心库
    opencv_videoio
    opencv_imgcodecs
    opencv_imgproc
    opencv_core

    # OpenCV依赖库（完全匹配你找到的文件名）
    csi_cv
    libpng
    libjpeg-turbo
    libtiff
    libwebp
    libopenjp2
    zlib

    # 系统库（静态链接必须）
    pthread
    dl
    m
    stdc++
)

target_link_libraries(${split_launcher_bin}
    pthread
    dl
    m
    stdc++
)

install(TARGETS ${bin} ${ai_event_bin} ${video_display_bin} ${split_launcher_bin} DESTINATION falldown_detect_elf)
