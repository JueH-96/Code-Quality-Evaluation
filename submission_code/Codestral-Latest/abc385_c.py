import sys
from collections import defaultdict

# Read input
N = int(sys.stdin.readline().strip())
H = list(map(int, sys.stdin.readline().strip().split()))

# Dictionary to store the frequency of each height
height_freq = defaultdict(int)

# Dictionary to store the positions of each height
height_positions = defaultdict(list)

# Populate the dictionaries
for i in range(N):
    height_freq[H[i]] += 1
    height_positions[H[i]].append(i)

# Function to check if a list of positions forms an arithmetic sequence
def is_arithmetic_sequence(positions):
    if len(positions) < 2:
        return True
    diff = positions[1] - positions[0]
    for i in range(2, len(positions)):
        if positions[i] - positions[i-1] != diff:
            return False
    return True

# Find the maximum number of buildings that can be chosen
max_buildings = 1
for height, freq in height_freq.items():
    if freq > max_buildings:
        positions = height_positions[height]
        if is_arithmetic_sequence(positions):
            max_buildings = freq

# Print the result
print(max_buildings)