N = int(input().strip())
grid = [list(map(int, list(input().strip()))) for _ in range(N)]

new_grid = [row.copy() for row in grid]

# top row
new_grid[0][1:] = grid[0][:-1]

# bottom row
new_grid[-1][:-1] = grid[-1][1:]

# left column
for i in range(1, N):
    new_grid[i][0] = grid[i-1][0]

# right column
for i in range(N-1):
    new_grid[i][-1] = grid[i+1][-1]

for row in new_grid:
    print(''.join(map(str, row)))