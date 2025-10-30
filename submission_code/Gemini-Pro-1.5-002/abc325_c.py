# YOUR CODE HERE
def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    visited = [[False] * w for _ in range(h)]
    count = 0
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w
    
    def dfs(r, c):
        visited[r][c] = True
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc) and grid[nr][nc] == '#' and not visited[nr][nc]:
                    dfs(nr, nc)
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '#' and not visited[r][c]:
                count += 1
                dfs(r, c)
                
    print(count)

solve()