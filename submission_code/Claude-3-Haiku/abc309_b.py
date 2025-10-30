# Read the input
N = int(input())
grid = []
for _ in range(N):
    row = list(map(int, input().split()))
    grid.append(row)

# Shift the outer squares clockwise
new_grid = [row[:] for row in grid]
for i in range(N):
    for j in range(N):
        if i == 0 or i == N-1 or j == 0 or j == N-1:
            if i == 0 and j == 0:
                new_grid[i][j] = grid[i+1][j]
            elif i == 0 and j == N-1:
                new_grid[i][j] = grid[i][j-1]
            elif i == N-1 and j == N-1:
                new_grid[i][j] = grid[i-1][j]
            elif i == N-1 and j == 0:
                new_grid[i][j] = grid[i][j+1]
            elif i == 0:
                new_grid[i][j] = grid[i][j-1]
            elif i == N-1:
                new_grid[i][j] = grid[i][j+1]
            elif j == 0:
                new_grid[i][j] = grid[i+1][j]
            elif j == N-1:
                new_grid[i][j] = grid[i-1][j]

# Print the result
for row in new_grid:
    print(''.join(map(str, row)))