import sys
import bisect

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0
Q = int(data[index])
index += 1

# Initialize variables
global_time = 0
start_idx = 0
time_list = []
count_list = []

# Process each query
for _ in range(Q):
    query_type = int(data[index])
    index += 1
    if query_type == 1:
        # Add a new plant with height 0 at current global time
        if not time_list or time_list[-1] != global_time:
            time_list.append(global_time)
            count_list.append(1)
        else:
            count_list[-1] += 1
    elif query_type == 2:
        # Wait for T days, increase global time by T
        T = int(data[index])
        index += 1
        global_time += T
    elif query_type == 3:
        # Harvest plants with height at least H
        H = int(data[index])
        index += 1
        thresh = global_time - H
        # Find the rightmost index where time <= thresh using binary search
        idx = bisect.bisect_right(time_list, thresh, lo=start_idx)
        if idx > start_idx:
            # Calculate the number of harvested plants
            harvested = 0
            for j in range(start_idx, idx):
                harvested += count_list[j]
            # Output the number harvested
            print(harvested)
            # Update start index to remove harvested plants
            start_idx = idx
        else:
            # No plants harvested
            print(0)