CXXFLAGS = -O3 # Optimization flags
LFLAGS = -ltbb # Library flags

all:
	g++ ${LFLAGS} ${CXXFLAGS} -o bin/kmeans src/kmeans-serial.cpp
