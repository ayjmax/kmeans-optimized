CXXFLAGS = -O3 # Optimization flags
LFLAGS = -L oneapi-tbb-2022.0.0/lib/intel64/gcc4.8 -ltbb # Library flags
IFLAGS = -Ioneapi-tbb-2022.0.0/include
# SFLAG= -fsanitize=address # not an option??? causes bugs when using this flag

all: serial serial-fast serial-fast-unroll parallel-simple parallel-fast

serial:
	g++ ${CXXFLAGS} ${SFLAG} -o bin/kmeans-serial src/kmeans-serial.cpp

serial-fast:
	g++ ${CXXFLAGS} ${SFLAG} -o bin/kmeans-serial-fast src/kmeans-serial-fast.cpp

serial-fast-unroll:
	g++ ${CXXFLAGS} ${SFLAG} -o bin/kmeans-serial-fast-unroll src/kmeans-serial-fast-unroll.cpp

parallel-simple:
	g++ ${CXXFLAGS} ${SFLAG} ${IFLAGS} -o bin/kmeans-parallel-simple src/kmeans-parallel-simple.cpp ${LFLAGS}

parallel-fast:
	g++ ${CXXFLAGS} ${SFLAG} ${IFLAGS} -o bin/kmeans-parallel-fast src/kmeans-parallel-fast.cpp ${LFLAGS}

clean:
	rm -r bin/*
