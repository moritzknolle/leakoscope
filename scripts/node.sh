#!/bin/bash

# Usage message
usage() {
    echo "Usage: $0 [-p <partition1,partition2,...>]"
    echo "If no partitions are specified, all available partitions will be checked."
    exit 1
}

# Function to get all available partitions
get_all_partitions() {
    sinfo -h -o "%R" | sort -u | tr '\n' ',' | sed 's/,$//'
}

# Parse arguments
while getopts ":p:" opt; do
    case ${opt} in
        p)
            PARTITIONS=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" 1>&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." 1>&2
            usage
            ;;
    esac
done

# If no partitions are specified, get all available partitions
if [ -z "$PARTITIONS" ]; then
    PARTITIONS=$(get_all_partitions)
fi

# Convert comma-separated partitions into a space-separated list
PARTITION_LIST=$(echo $PARTITIONS | tr ',' ' ')

# Function to extract the relevant information for each node
extract_gpu_info() {
    NODE=$1
    # Extract node information using scontrol
    NODE_INFO=$(scontrol show node $NODE)

    # Parse the node's CPU, GPU, and memory usage
    CPU_USED=$(echo "$NODE_INFO" | grep -oP 'CPUAlloc=\K\d+')
    CPU_TOTAL=$(echo "$NODE_INFO" | grep -oP 'CPUTot=\K\d+')

    # Extract GPU info from AllocTRES and CfgTRES fields
    GPU_USED=$(echo "$NODE_INFO" | grep -oP 'AllocTRES=.*gres/gpu=\K\d+' || echo "N/A")
    GPU_TOTAL=$(echo "$NODE_INFO" | grep -oP 'CfgTRES=.*gres/gpu=\K\d+' || echo "N/A")

    MEM_USED=$(echo "$NODE_INFO" | grep -oP 'AllocMem=\K\d+')
    MEM_TOTAL=$(echo "$NODE_INFO" | grep -oP 'RealMemory=\K\d+')

    # Convert memory usage from MB to GB
    MEM_USED_GB=$(echo "$MEM_USED/1024" | bc)
    MEM_TOTAL_GB=$(echo "$MEM_TOTAL/1024" | bc)

    # Find the users using the node (if any)
    USERS=$(squeue -h -w $NODE -o %u | sort | uniq | tr '\n' ',' | sed 's/,$//')

    # Default GPU values to "N/A" if they are empty or unavailable
    GPU_USED=${GPU_USED:-"N/A"}
    GPU_TOTAL=${GPU_TOTAL:-"N/A"}
    USERS=${USERS:-"None"}

    # Print the summary for the node
    echo "$NODE: CPU: $CPU_USED/$CPU_TOTAL, GPU: $GPU_USED/$GPU_TOTAL, Mem: ${MEM_USED_GB}GB/${MEM_TOTAL_GB}GB, Users: [$USERS]"
}

# Loop through the partitions and fetch the node information
for PARTITION in $PARTITION_LIST; do
    echo "Node Summary for Partition: $PARTITION"
    echo "--------------------------------------------"
    echo "Node | CPU Usage | GPU Usage | Memory Usage | Users"

    # Fetch node information for the partition
    NODES=$(sinfo -N -p $PARTITION --noheader | awk '{print $1}')
    
    # Loop through all nodes in the partition and extract GPU info
    for NODE in $NODES; do
        extract_gpu_info $NODE
    done
    echo ""
done 