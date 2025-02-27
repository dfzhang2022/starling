# usage: bash check_num.sh {numa_node_num}
for device in /sys/bus/pci/devices/*; do
    if [ -f "$device/numa_node" ]; then
        numa_node=$(cat "$device/numa_node")
        if [ "$numa_node" -eq $1 ]; then
            pci_id=$(basename "$device")
            lspci -s "$pci_id"
        fi
    fi
done