# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

# Create a dictionary to store the indices of each color
color_indices = {}
for i, color in enumerate(A):
    if color not in color_indices:
        color_indices[color] = [i]
    else:
        color_indices[color].append(i)

# Count the number of colors that have exactly one person in between two people wearing the same color
count = 0
for indices in color_indices.values():
    if abs(indices[0] - indices[1]) == 2:
        count += 1

print(count)