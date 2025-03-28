DATASET=datasets/bean.txt

source oneapi-tbb-2022.0.0/env/vars.sh

make clean
make all

echo "" > output.txt

# echo "------------------------- Serial -------------------------" > output.txt
# cat ${DATASET} | bin/kmeans-serial >> output.txt

# echo "------------------------- Serial Fast -------------------------" >> output.txt
# cat ${DATASET} | bin/kmeans-serial-fast >> output.txt

# echo "------------------------- Serial Fast Unroll -------------------------" >> output.txt
# cat ${DATASET} | bin/kmeans-serial-fast-unroll >> output.txt

# echo "------------------------- Serial Fast No Cluster -------------------------" >> output.txt
# cat ${DATASET} | bin/kmeans-serial-fast-no-cluster >> output.txt

echo "------------------------- Parallel Simple -------------------------" >> output.txt
cat ${DATASET} | bin/kmeans-parallel-simple >> output.txt

echo "------------------------- Parallel Fast -------------------------" >> output.txt
cat ${DATASET} | bin/kmeans-parallel-fast >> output.txt