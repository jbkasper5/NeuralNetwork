SRCS=$(shell find $(PWD) -name \*.cpp)
OBJS=$(SRCS:%.cpp=%.o)

CC=g++
OPT=-O3
CPPFLAGS=$(pkg-config --cflags --libs opencv4) -std=c++11 -I/usr/local/include/opencv4

TARGET=net

$(TARGET): $(OBJS)
	$(CC) $^ -o $@ $(CPPFLAGS)

%.o: %.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

.PHONY: clean
clean:
	$(RM) -rf $(TARGET)
	$(RM) -rf $(OBJS)

.PHONY: print
print:
	echo $(pkg-config --cflags opencv4)
	echo $(pkg-config --libs opencv4)
