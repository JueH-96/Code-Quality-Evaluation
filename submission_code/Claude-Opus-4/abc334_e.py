def find_components(grid, H, W):
    visited = [[False] * W for _ in range(H)]
    components = []
    component_id = [[None] * W for _ in range(H)]
    
    def dfs(i, j, comp_id):
        if i < 0 or i >= H or j < 0 or j >= W:
            return
        if visited[i][j] or grid[i][j] == '.':
            return
        
        visited[i][j] = True
        component_id[i][j] = comp_id
        
        # Check 4 adjacent cells
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(i + di, j + dj, comp_id)
    
    comp_count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                dfs(i, j, comp_count)
                comp_count += 1
    
    return comp_count, component_id

def mod_inverse(a, mod):
    return pow(a, mod - 2, mod)

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find current green components
current_components, component_id = find_components(grid, H, W)

# Count red cells and calculate sum of components after repainting
red_cells = []
total_components = 0

for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            red_cells.append((i, j))
            
            # Check adjacent green cells
            adjacent_components = set()
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                    adjacent_components.add(component_id[ni][nj])
            
            # Calculate components after repainting this cell
            if len(adjacent_components) == 0:
                # Not adjacent to any green cells, adds 1 new component
                total_components += current_components + 1
            else:
                # Merges k components into 1
                total_components += current_components - len(adjacent_components) + 1

# Calculate expected value modulo 998244353
MOD = 998244353
num_red_cells = len(red_cells)

# Expected value = total_components / num_red_cells
# We need to find R such that R * num_red_cells ≡ total_components (mod MOD)
# So R = total_components * inverse(num_red_cells) (mod MOD)

result = (total_components * mod_inverse(num_red_cells, MOD)) % MOD
print(result)