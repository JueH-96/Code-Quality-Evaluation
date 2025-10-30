def count_components(grid, H, W):
    visited = [[False] * W for _ in range(H)]
    components = 0
    
    def dfs(i, j):
        if i < 0 or i >= H or j < 0 or j >= W or visited[i][j] or grid[i][j] == '.':
            return
        visited[i][j] = True
        # Check 4 directions
        dfs(i+1, j)
        dfs(i-1, j)
        dfs(i, j+1)
        dfs(i, j-1)
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                dfs(i, j)
                components += 1
    
    return components

def mod_inverse(a, mod):
    # Extended Euclidean Algorithm to find modular inverse
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a % mod, mod)
    return (x % mod + mod) % mod

# Read input
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(input().strip()))

# Find all red cells
red_cells = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            red_cells.append((i, j))

total_components = 0
num_red_cells = len(red_cells)

# For each red cell, simulate repainting it green
for i, j in red_cells:
    # Temporarily make it green
    grid[i][j] = '#'
    # Count components
    components = count_components(grid, H, W)
    total_components += components
    # Restore to red
    grid[i][j] = '.'

# Expected value is total_components / num_red_cells
# We need to find R such that R * num_red_cells ≡ total_components (mod 998244353)

MOD = 998244353
numerator = total_components % MOD
denominator = num_red_cells % MOD

# Find modular inverse of denominator
inv_denominator = mod_inverse(denominator, MOD)

# Calculate result
result = (numerator * inv_denominator) % MOD
print(result)