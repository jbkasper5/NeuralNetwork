SRCS=$(shell find $(PWD) -name \*.cpp)
OBJS=$(SRCS:%.cpp=%.o)

CC=g++
OPT=-O3
CPPFLAGS=-std=c++11 `pkg-config --cflags opencv4`
LDFLAGS=`pkg-config --libs opencv4`

TARGET=net

$(TARGET): $(OBJS)
	$(CC) $^ -o $@ $(CPPFLAGS) $(LDFLAGS)

%.o: %.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

.PHONY: clean
clean:
	$(RM) -rf $(TARGET)
	$(RM) -rf $(OBJS)

