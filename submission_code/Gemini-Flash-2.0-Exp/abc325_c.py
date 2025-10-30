def solve():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    
    def get_neighbors(r, c):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    neighbors.append((nr, nc))
        return neighbors

    def dfs(r, c, visited, components):
        visited[r][c] = True
        components.append((r, c))
        
        for nr, nc in get_neighbors(r, c):
            if grid[nr][nc] == '#' and not visited[nr][nc]:
                dfs(nr, nc, visited, components)

    visited = [[False] * W for _ in range(H)]
    num_sensors = 0
    
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '#' and not visited[r][c]:
                num_sensors += 1
                components = []
                dfs(r, c, visited, components)

    print(num_sensors)

solve()