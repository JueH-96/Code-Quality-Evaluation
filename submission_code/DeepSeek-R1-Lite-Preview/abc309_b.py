N = int(input())
grid = [list(input().strip()) for _ in range(N)]

# Collect perimeter values in clockwise order
perimeter = []

# Top row: left to right
for j in range(N):
    perimeter.append(grid[0][j])

# Right column: top+1 to bottom
for i in range(1, N):
    perimeter.append(grid[i][N-1])

# Bottom row: right-1 to left
for j in range(N-2, -1, -1):
    perimeter.append(grid[N-1][j])

# Left column: bottom-1 to top+1
for i in range(N-2, 0, -1):
    perimeter.append(grid[i][0])

# Shift perimeter values clockwise by one
shifted_perimeter = perimeter[1:] + [perimeter[0]]

# Place shifted values back into the grid
index = 0

# Top row: left to right
for j in range(N):
    grid[0][j] = shifted_perimeter[index]
    index += 1

# Right column: top+1 to bottom
for i in range(1, N):
    grid[i][N-1] = shifted_perimeter[index]
    index += 1

# Bottom row: right-1 to left
for j in range(N-2, -1, -1):
    grid[N-1][j] = shifted_perimeter[index]
    index += 1

# Left column: bottom-1 to top+1
for i in range(N-2, 0, -1):
    grid[i][0] = shifted_perimeter[index]
    index += 1

# Print the updated grid
for row in grid:
    print(''.join(row))