#!/bin/bash

# 定义要禁用的核心范围
ranges=("192-287" "288-383")
echo "Reallllllly BE CARE for what you are doing"
echo "Blocking cores on"
for range in "${ranges[@]}"; do
    echo ${range}
done

# 遍历每个范围
for range in "${ranges[@]}"; do
    # 分割范围为起始和结束核心编号
    start=$(echo "$range" | cut -d '-' -f 1)
    end=$(echo "$range" | cut -d '-' -f 2)

    # 遍历范围内的每个核心编号
    for ((core = start; core <= end; core++)); do
        # 检查核心对应的 online 文件是否存在
        if [ -f "/sys/devices/system/cpu/cpu$core/online" ]; then
            # 禁用核心
            # echo 1 | sudo tee /sys/devices/system/cpu/cpu$core/online
            echo "echo 0 | sudo tee /sys/devices/system/cpu/cpu$core/online"
            echo "Disabled CPU core $core"
        else
            echo "CPU core $core does not exist or is not controllable."
        fi
    done
done