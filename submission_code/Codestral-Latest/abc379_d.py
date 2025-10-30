import sys
from collections import deque

input = sys.stdin.read
data = input().split()

Q = int(data[0])
index = 1

# Initialize variables
plants = deque()
total_height = 0
harvested_counts = []

# Process each query
for _ in range(Q):
    query_type = int(data[index])
    index += 1

    if query_type == 1:
        # Prepare one empty flower pot and put a plant in it
        plants.append(0)

    elif query_type == 2:
        # Wait for T days
        T = int(data[index])
        index += 1
        total_height += T

    elif query_type == 3:
        # Harvest all plants with a height of at least H
        H = int(data[index])
        index += 1
        harvest_height = H - total_height

        # Count and remove plants with height at least H
        count = 0
        while plants and plants[0] <= harvest_height:
            plants.popleft()
            count += 1

        harvested_counts.append(count)

# Output the results
for count in harvested_counts:
    print(count)