def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    snuke = "snuke"
    
    q = [((0, 0), 0)]
    visited = set()
    
    while q:
        (r, c), step = q.pop(0)
        
        if (r, c) == (h - 1, w - 1) and grid[r][c] == snuke[step % 5]:
            print("Yes")
            return
        
        if (r, c, step) in visited:
            continue
        visited.add((r, c, step))
        
        if grid[r][c] != snuke[step % 5]:
            continue
        
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                q.append(((nr, nc), step + 1))
    
    print("No")

solve()