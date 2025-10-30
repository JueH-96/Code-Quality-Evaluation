# YOUR CODE HERE
N = int(input())
grid = [list(map(int, list(input().strip()))) for _ in range(N)]

# Extract the outer layer in clockwise order
outer = []
for j in range(N):
    outer.append(grid[0][j])
for i in range(1, N):
    outer.append(grid[i][N-1])
for j in range(N-2, -1, -1):
    outer.append(grid[N-1][j])
for i in range(N-2, 0, -1):
    outer.append(grid[i][0])

# Shift the outer list clockwise by one
shifted_outer = [outer[-1]] + outer[:-1]

# Reconstruct the grid with the shifted outer layer
index = 0
for j in range(N):
    grid[0][j] = shifted_outer[index]
    index += 1
for i in range(1, N):
    grid[i][N-1] = shifted_outer[index]
    index += 1
for j in range(N-2, -1, -1):
    grid[N-1][j] = shifted_outer[index]
    index += 1
for i in range(N-2, 0, -1):
    grid[i][0] = shifted_outer[index]
    index += 1

# Print the resulting grid
for row in grid:
    print(''.join(map(str, row)))