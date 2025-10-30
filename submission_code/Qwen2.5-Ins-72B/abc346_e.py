# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

H, W, M = map(int, data[:3])
operations = [(int(data[3*i]), int(data[3*i+1]), int(data[3*i+2])) for i in range(M)]

row_colors = defaultdict(int)
col_colors = defaultdict(int)
row_operations = defaultdict(list)
col_operations = defaultdict(list)

for T, A, X in operations:
    if T == 1:
        row_operations[A].append(X)
    else:
        col_operations[A].append(X)

for row, colors in row_operations.items():
    row_colors[colors[-1]] += W

for col, colors in col_operations.items():
    col_colors[colors[-1]] += H

# Adjust for overlaps
for row, colors in row_operations.items():
    for col, colors2 in col_operations.items():
        if colors[-1] == colors2[-1]:
            row_colors[colors[-1]] -= 1
        else:
            col_colors[colors2[-1]] -= 1

color_count = defaultdict(int)
for color, count in row_colors.items():
    color_count[color] += count
for color, count in col_colors.items():
    color_count[color] += count

K = len(color_count)
print(K)
for color in sorted(color_count.keys()):
    print(color, color_count[color])