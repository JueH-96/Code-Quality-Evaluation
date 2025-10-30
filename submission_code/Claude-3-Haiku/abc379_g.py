# Read the input
H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]

# Count the number of '?' in the grid
q = sum(row.count('?') for row in grid)

# Define a function to check if a grid satisfies the condition
def is_valid(grid):
    for i in range(H):
        for j in range(W):
            if j < W - 1 and grid[i][j] == grid[i][j+1]:
                return False
            if i < H - 1 and grid[i][j] == grid[i+1][j]:
                return False
    return True

# Count the number of valid grids
count = 0
for mask in range(3 ** q):
    new_grid = [row[:] for row in grid]
    idx = 0
    for i in range(H):
        for j in range(W):
            if new_grid[i][j] == '?':
                new_grid[i][j] = str(((mask // 3 ** idx) % 3) + 1)
                idx += 1
    if is_valid(new_grid):
        count += 1

# Print the answer modulo 998244353
print(count % 998244353)