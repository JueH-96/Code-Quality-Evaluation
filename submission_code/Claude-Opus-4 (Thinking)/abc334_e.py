def find_components(grid):
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    component_id = [[-1] * W for _ in range(H)]
    num_components = 0
    
    def dfs(i, j, comp_id):
        if i < 0 or i >= H or j < 0 or j >= W:
            return
        if visited[i][j] or grid[i][j] == '.':
            return
        
        visited[i][j] = True
        component_id[i][j] = comp_id
        
        # Check all 4 neighbors
        dfs(i-1, j, comp_id)
        dfs(i+1, j, comp_id)
        dfs(i, j-1, comp_id)
        dfs(i, j+1, comp_id)
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                dfs(i, j, num_components)
                num_components += 1
    
    return num_components, component_id

def get_adjacent_components(i, j, component_id, grid):
    H, W = len(grid), len(grid[0])
    adjacent_comps = set()
    
    # Check all 4 neighbors
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
            adjacent_comps.add(component_id[ni][nj])
    
    return len(adjacent_comps)

def mod_inverse(a, mod):
    return pow(a, mod - 2, mod)

def main():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(input().strip())
    
    # Find current number of green components
    current_components, component_id = find_components(grid)
    
    # Count red cells and calculate sum of components after each repainting
    red_cells = 0
    sum_components = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_cells += 1
                adj_comps = get_adjacent_components(i, j, component_id, grid)
                
                if adj_comps == 0:
                    # Creates a new component
                    sum_components += current_components + 1
                elif adj_comps == 1:
                    # Joins an existing component, no change
                    sum_components += current_components
                else:
                    # Merges adj_comps components into 1
                    sum_components += current_components - adj_comps + 1
    
    # Expected value = sum_components / red_cells
    MOD = 998244353
    
    # Result = sum_components * inverse(red_cells) mod MOD
    result = (sum_components * mod_inverse(red_cells, MOD)) % MOD
    
    print(result)

if __name__ == "__main__":
    main()