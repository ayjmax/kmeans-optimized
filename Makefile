CXXFLAGS = -O3 # Optimization flags
LFLAGS = -ltbb # Library flags
VERSION = 3

all:
	g++ ${LFLAGS} ${CXXFLAGS} -o bin/kmeans hw2/kmeans-serial.cpp
	source oneapi-tbb-2022.0.0/env/vars.sh
	cat datasets/XXXX.txt | ./kmeans