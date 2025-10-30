# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
grid = [list(map(int, data[i+1])) for i in range(N)]

# Extract outer squares
top_row = grid[0][1:N-1]
right_col = [grid[i][N-1] for i in range(1, N-1)]
bottom_row = grid[N-1][N-2::-1]
left_col = [grid[i][0] for i in range(N-2, 0, -1)]

# Shift outer squares clockwise
top_row = [top_row[-1]] + top_row[:-1]
right_col = [right_col[-1]] + right_col[:-1]
bottom_row = [bottom_row[-1]] + bottom_row[:-1]
left_col = [left_col[-1]] + left_col[:-1]

# Update grid
for i in range(1, N-1):
    grid[0][i] = top_row[i-1]
    grid[i][N-1] = right_col[i-1]
    grid[N-1][N-1-i] = bottom_row[i-1]
    grid[N-1-i][0] = left_col[i-1]

# Print resulting grid
for row in grid:
    print(''.join(map(str, row)))