# Read the input
N = int(input())

# Initialize the grid
grid = [['.' for _ in range(N)] for _ in range(N)]

# Apply the coloring operations
for i in range(1, N+1):
    j = N + 1 - i
    if i <= j:
        for x in range(i-1, j):
            for y in range(i-1, j):
                if (i % 2) == 1:
                    grid[x][y] = '#'
                else:
                    grid[x][y] = '.'

# Print the final grid
for row in grid:
    print(''.join(row))