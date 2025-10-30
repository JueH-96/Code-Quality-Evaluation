import sys
from collections import defaultdict

n = int(sys.stdin.readline())
a = list(map(int, sys.stdin.readline().split()))

# Create a dictionary to store the positions of each color
color_positions = defaultdict(list)

# Record the 1-based positions for each color
for index, color in enumerate(a):
    color_positions[color].append(index + 1)

count = 0

# Check each color from 1 to N
for color in range(1, n + 1):
    positions = color_positions[color]
    # Calculate the absolute difference between the two positions
    if abs(positions[0] - positions[1]) == 2:
        count += 1

print(count)