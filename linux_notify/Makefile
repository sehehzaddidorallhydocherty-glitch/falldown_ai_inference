K230_SDK_ROOT ?= /home/ubuntu/k230_sdk
CDK_SRC_DIR ?= $(K230_SDK_ROOT)/src/common/cdk
MPP_SRC_DIR ?= $(K230_SDK_ROOT)/src/big/mpp
IPCMSG_ROOT := $(CDK_SRC_DIR)/user/component/ipcmsg
DATAFIFO_ROOT := $(CDK_SRC_DIR)/user/component/datafifo
MAPI_ROOT := $(CDK_SRC_DIR)/user/mapi

LINUX_TOOLCHAIN_DIR ?= $(K230_SDK_ROOT)/toolchain/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.0/bin
CXX := $(LINUX_TOOLCHAIN_DIR)/riscv64-unknown-linux-gnu-g++
BIN ?= falldown_notify_client

ROOT_DIR := ..
COMM_DIR := $(ROOT_DIR)/third_party/k230_comm

SRCS := falldown_notify_client.cpp
SRCS += $(COMM_DIR)/client/comm_client.cpp
SRCS += $(COMM_DIR)/client/event_reader.cpp
SRCS += $(COMM_DIR)/client/avdata_reader.cpp
SRCS += $(COMM_DIR)/client/avdata_writer.cpp
SRCS += $(COMM_DIR)/client/data_fifo_reader.cpp
SRCS += $(COMM_DIR)/client/data_fifo_writer.cpp
SRCS += $(COMM_DIR)/client/data_shm.cpp

OBJS := $(SRCS:.cpp=.o)

INCS := -I$(COMM_DIR)/include
INCS += -I$(COMM_DIR)/client
INCS += -I$(MPP_SRC_DIR)/include
INCS += -I$(MPP_SRC_DIR)/include/comm
INCS += -I$(MPP_SRC_DIR)/userapps/api
INCS += -I$(IPCMSG_ROOT)/include
INCS += -I$(DATAFIFO_ROOT)/include
INCS += -I$(MAPI_ROOT)/include
INCS += -I$(MAPI_ROOT)/include/api
INCS += -I$(MAPI_ROOT)/include/comm

LIBPATH := -L$(IPCMSG_ROOT)/host/lib
LIBPATH += -L$(DATAFIFO_ROOT)/slave/lib
LIBPATH += -L$(MAPI_ROOT)/mediaclient/lib

LIBS := -Wl,--start-group -lmapi -ldatafifo -lipcmsg -lpthread -Wl,--end-group

CXXFLAGS += -std=c++17 -O2

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LIBPATH) $(LIBS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCS) $< -o $@

clean:
	rm -f $(BIN) $(OBJS)
