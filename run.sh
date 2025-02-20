DATASET=datasets/bean.txt
# DATASET=datasets/dataset2.txt

source oneapi-tbb-2022.0.0/env/vars.sh

make all

cat ${DATASET} | bin/kmeans > output.txt