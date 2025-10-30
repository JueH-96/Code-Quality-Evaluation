import sys
from collections import defaultdict

def max_buildings_with_same_height_and_equal_intervals(N, H):
    # Dictionary to store the indices of buildings with the same height
    height_indices = defaultdict(list)
    for i, height in enumerate(H):
        height_indices[height].append(i)

    # Function to calculate the maximum number of buildings with equal intervals
    def max_equal_interval_count(indices):
        n = len(indices)
        if n == 1:
            return 1

        max_count = 1
        for step in range(1, n):
            count = 1
            current = indices[0]
            for i in range(1, n):
                if indices[i] - current == step:
                    count += 1
                    current = indices[i]
                elif indices[i] - current > step:
                    break
            max_count = max(max_count, count)

        return max_count

    # Calculate the maximum number of buildings for each height
    max_buildings = 1
    for height in height_indices:
        max_buildings = max(max_buildings, max_equal_interval_count(height_indices[height]))

    return max_buildings

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
H = list(map(int, data[1:]))

# Calculate and print the result
result = max_buildings_with_same_height_and_equal_intervals(N, H)
print(result)