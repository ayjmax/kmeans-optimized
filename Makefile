CXXFLAGS = -O3 # Optimization flags
LFLAGS = -L oneapi-tbb-2022.0.0/lib/intel64/gcc4.8 -ltbb # Library flags
IFLAGS = -Ioneapi-tbb-2022.0.0/include
# SFLAG= -fsanitize=address # not an option??? causes bugs when using this flag

all: serial serial-fast parallel-simple parallel-tbb

serial:
	g++ ${CXXFLAGS} ${SFLAG} -o bin/kmeans-serial src/kmeans-serial.cpp

serial-fast:
	g++ ${CXXFLAGS} ${SFLAG} -o bin/kmeans-serial-fast src/kmeans-serial-fast.cpp

parallel-simple:
	g++ ${CXXFLAGS} ${SFLAG} ${IFLAGS} -o bin/kmeans-parallel-simple src/kmeans-parallel-simple.cpp ${LFLAGS}

parallel-tbb:
	g++ ${CXXFLAGS} ${SFLAG} ${IFLAGS} -o bin/kmeans-parallel-tbb src/kmeans-parallel-tbb.cpp ${LFLAGS}

clean:
	rm -r bin/*
