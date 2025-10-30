import sys

def solve():
    h, w = map(int, sys.stdin.readline().split())
    grid_str = [sys.stdin.readline().strip() for _ in range(h)]
    is_sensor = [[False for _ in range(w)] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if grid_str[i][j] == '#':
                is_sensor[i][j] = True
    
    visited_grid = [[False for _ in range(w)] for _ in range(h)]
    component_count = 0
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w
        
    def dfs(r, c):
        visited_grid[r][c] = True
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc) and is_sensor[nr][nc] and not visited_grid[nr][nc]:
                    dfs(nr, nc)
                    
    for i in range(h):
        for j in range(w):
            if is_sensor[i][j] and not visited_grid[i][j]:
                component_count += 1
                dfs(i, j)
                
    print(component_count)

if __name__ == '__main__':
    solve()