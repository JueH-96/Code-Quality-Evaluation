def find_connected_sensors(grid, H, W):
    def dfs(i, j, visited):
        if i < 0 or i >= H or j < 0 or j >= W or grid[i][j] == '.' or (i,j) in visited:
            return
        
        visited.add((i,j))
        
        # Check all 8 adjacent cells
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                dfs(ni, nj, visited)

    visited = set()
    count = 0
    
    # Iterate through each cell
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and (i,j) not in visited:
                dfs(i, j, visited)
                count += 1
                
    return count

# Read input
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())

# Print answer
print(find_connected_sensors(grid, H, W))