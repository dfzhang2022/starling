mkdir -p ./bigann && cd ./bigann

axel -n 5  https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin

axel -n 5 https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/learn.100M.u8bin

axel -n 5 https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin


cd ../


mkdir -p ./DEEP && cd ./DEEP
axel -n 5 https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/learn.350M.fbin

axel -n 5 https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin

axel -n 5 https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin

cd ../
