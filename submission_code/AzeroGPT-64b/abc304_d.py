from bisect import bisect_left, bisect_right
import numpy as np

# Reading the inputs from stdin
W, H = map(int, input().split())
N = int(input())
points = [tuple(map(int, input().split())) for _ in range(N)]
A = int(input())
x_cuts = list(map(int, input().split()))
B = int(input())
y_cuts = list(map(int, input().split()))

# Precomputing the positions of cuts with respect to points in x and y directions
x_positions = [0] + [bisect_left(x_cuts, p[0]) for p in points] + [A + 1]
y_positions = [0] + [bisect_left(y_cuts, p[1]) for p in points] + [B + 1]

# Counting the occurrence of each (x, y) combination of positions
x_counts = np.bincount(x_positions)
y_counts = np.bincount(y_positions)

# Calculating the total number of points in each (x, y) bin
x_binned = np.cumsum(x_counts)
y_binned = np.cumsum(y_counts)

# Determining the maximum number of points in any (A+1) * (B+1) bin
max_points = 0
for i in range(A + 1):
    for j in range(B + 1):
        slice_x = slice(x_binned[i], x_binned[i + 1])
        slice_y = slice(y_binned[j], y_binned[j + 1])
        max_points = max(max_points, x_binned[i + 1] - x_binned[i] + y_binned[j + 1] - y_binned[j] - N)

# The minimum number of points in a bin is when the points are evenly distributed
min_points = 0 if N % ((A + 1) * (B + 1)) == 0 else 1

# Printing the result to stdout
print(f"{min_points} {max_points}")