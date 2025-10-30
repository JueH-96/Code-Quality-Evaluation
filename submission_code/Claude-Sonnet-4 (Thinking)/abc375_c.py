n = int(input())
grid = []
for _ in range(n):
    grid.append(list(input().strip()))

for i in range(1, n // 2 + 1):
    new_grid = [row[:] for row in grid]  # Create a copy
    
    for x in range(i, n + 1 - i + 1):
        for y in range(i, n + 1 - i + 1):
            # Replace cell (y, N + 1 - x) with cell (x, y)
            # Converting to 0-based indexing:
            new_grid[y-1][n - x] = grid[x-1][y-1]
    
    grid = new_grid

for row in grid:
    print(''.join(row))