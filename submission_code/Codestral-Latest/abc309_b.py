# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
grid = [list(map(int, list(data[i]))) for i in range(1, N+1)]

# Extract the outer layer
top = grid[0][1:-1]
right = [grid[i][-1] for i in range(1, N-1)]
bottom = grid[-1][1:-1][::-1]
left = [grid[i][0] for i in range(N-2, 0, -1)]

# Rotate the outer layer
rotated_top = [left[0]] + top
rotated_right = right + [bottom[-1]]
rotated_bottom = [top[-1]] + bottom[:-1]
rotated_left = [left[1:]] + [rotated_top[0]]

# Place the rotated outer layer back into the grid
grid[0][1:-1] = rotated_top
for i in range(1, N-1):
    grid[i][-1] = rotated_right[i-1]
grid[-1][1:-1] = rotated_bottom[::-1]
for i in range(1, N-1):
    grid[i][0] = rotated_left[i-1]

# Print the resulting grid
for row in grid:
    print(''.join(map(str, row)))