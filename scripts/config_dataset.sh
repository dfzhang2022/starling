#!/bin/sh

# Switch dataset in the config_local.sh file by calling the desired function

#################
#   BIGANN10M   #
#################
dataset_bigann10M() {
  BASE_PATH=/data/dataset/BIGANN/base.10M.u8bin
  QUERY_FILE=/data/dataset/BIGANN/query.public.10K.128.u8bin
  GT_FILE=/data/dataset/BIGANN/bigann-10M-gt.bin 
  PREFIX=bigann_10m
  DATA_TYPE=uint8
  DIST_FN=l2
  B=0.3
  K=10
  DATA_DIM=128
  DATA_N=10000000
}
my_dataset_bigann10M() {
  BASE_PATH=/data/dataset/BIGANN/learn.100M.u8bin
  QUERY_FILE=/data/dataset/BIGANN/query.public.10K.u8bin
  GT_FILE=/data/dataset/BIGANN/GT_10M/bigann-10M
  PREFIX=bigann_100m
  DATA_TYPE=uint8
  DIST_FN=l2
  B=5
  K=10
  DATA_DIM=128
  DATA_N=10000000
}


##################
#   BIGANN100M   #
##################
dataset_bigann100M() {
  BASE_PATH=/data/dataset/bigann/learn.100M.u8bin
  QUERY_FILE=/data/dataset/bigann/query.public.10K.u8bin
  GT_FILE=/data/dataset/bigann/bigann-10M-gt.bin 
  PREFIX=bigann_100m
  DATA_TYPE=uint8
  DIST_FN=l2
  B=5
  K=100
  DATA_DIM=128
  DATA_N=10000000
}


################
#   BIGANN1B   #
################
dataset_bigann1B() {
  BASE_PATH=/data/dataset/bigann/base.1B.u8bin 
  QUERY_FILE=/data/dataset/bigann/query.public.10K.u8bin
  GT_FILE=/data/dataset/bigann/bigann-10M-gt.bin 
  PREFIX=bigann_100m
  DATA_TYPE=uint8
  DIST_FN=l2
  B=5
  K=100
  DATA_DIM=128
  DATA_N=10000000
}


####################
#     DEEP350M     #
####################


dataset_350M() {
  BASE_PATH=/data/dataset/DEEP/learn.350M.fbin
  QUERY_FILE=/data/dataset/DEEP/query.public.10K.fbin
  GT_FILE=/data/dataset/DEEP/deep-10k-gt.bin 
  PREFIX=bigann_100m
  DATA_TYPE=uint8
  DIST_FN=l2
  B=5
  K=100
  DATA_DIM=128
  DATA_N=10000000
}