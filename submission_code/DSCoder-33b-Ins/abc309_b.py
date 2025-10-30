# YOUR CODE HERE
N = int(input())
grid = [list(map(int, input())) for _ in range(N)]

# Shift the outer squares clockwise by one square each
for i in range(N):
    for j in range(N):
        if i == 0 or i == N-1 or j == 0 or j == N-1:
            if i == 0 and j == 0:
                grid[i][j] = grid[i+1][j]
            elif i == 0:
                grid[i][j] = grid[i][j-1]
            elif j == N-1:
                grid[i][j] = grid[i-1][j]
            elif i == N-1:
                grid[i][j] = grid[i][j+1]
            else:
                grid[i][j] = grid[i+1][j]

# Print the resulting grid
for row in grid:
    print(''.join(map(str, row)))