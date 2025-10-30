def solve():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    
    def is_valid(r, c):
        return 0 <= r < H and 0 <= c < W

    def can_move(r, c):
        if grid[r][c] == '#':
            return False
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and grid[nr][nc] == '#':
                return False
        return True

    def bfs(start_r, start_c):
        if not can_move(start_r, start_c):
            return 0
        
        q = [(start_r, start_c)]
        visited = {(start_r, start_c)}
        count = 1
        
        while q:
            r, c = q.pop(0)
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc) and (nr, nc) not in visited and can_move(nr, nc):
                    q.append((nr, nc))
                    visited.add((nr, nc))
                    count += 1
        return count

    max_freedom = 0
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '.':
                max_freedom = max(max_freedom, bfs(r, c))
    
    print(max_freedom)

solve()