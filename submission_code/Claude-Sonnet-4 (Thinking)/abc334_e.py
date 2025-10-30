H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(input().strip()))

def count_components():
    visited = [[False] * W for _ in range(H)]
    components = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                components += 1
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if x < 0 or x >= H or y < 0 or y >= W or visited[x][y] or grid[x][y] != '#':
                        continue
                    visited[x][y] = True
                    stack.append((x+1, y))
                    stack.append((x-1, y))
                    stack.append((x, y+1))
                    stack.append((x, y-1))
    
    return components

# Find all red cells
red_cells = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            red_cells.append((i, j))

total_components = 0

# For each red cell, simulate repainting it green
for r, c in red_cells:
    grid[r][c] = '#'
    total_components += count_components()
    grid[r][c] = '.'

# Calculate expected value and convert to modular arithmetic
num_red_cells = len(red_cells)
MOD = 998244353
inv = pow(num_red_cells, MOD-2, MOD)
result = (total_components * inv) % MOD

print(result)