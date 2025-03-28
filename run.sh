DATASET=datasets/bean.txt

source oneapi-tbb-2022.0.0/env/vars.sh

make clean
make all

echo "------------------------- Serial -------------------------" > output.txt
cat ${DATASET} | bin/kmeans-serial >> output.txt

echo "------------------------- Serial Fast -------------------------" >> output.txt
cat ${DATASET} | bin/kmeans-serial-fast >> output.txt

# echo "------------------------- Parallel Simple -------------------------" >> output.txt
# cat ${DATASET} | bin/kmeans-parallel-simple >> output.txt

# echo "------------------------- Parallel TBB -------------------------" >> output.txt
# cat ${DATASET} | bin/kmeans-parallel-tbb >> output.txt