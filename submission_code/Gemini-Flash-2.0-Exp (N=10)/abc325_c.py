def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    visited = [[False for _ in range(w)] for _ in range(h)]
    count = 0

    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w

    def dfs(r, c):
        if not is_valid(r, c) or visited[r][c] or grid[r][c] == '.':
            return
        
        visited[r][c] = True
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                dfs(r + dr, c + dc)

    for r in range(h):
        for c in range(w):
            if grid[r][c] == '#' and not visited[r][c]:
                dfs(r, c)
                count += 1
    
    print(count)

solve()