def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w
    
    def can_move_from(r, c):
        if grid[r][c] == '#':
            return False
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and grid[nr][nc] == '#':
                return False
        return True

    def bfs(start_r, start_c):
        if not can_move_from(start_r, start_c):
            return 1
        
        q = [(start_r, start_c)]
        visited = {(start_r, start_c)}
        count = 0
        
        while q:
            r, c = q.pop(0)
            count += 1
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc) and grid[nr][nc] == '.' and (nr, nc) not in visited and can_move_from(nr,nc):
                    
                    can_move = True
                    for dr2, dc2 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nnr, nnc = nr + dr2, nc + dc2
                        if is_valid(nnr, nnc) and grid[nnr][nnc] == '#':
                            can_move = False
                            break
                    if can_move:
                        q.append((nr, nc))
                        visited.add((nr, nc))
        return count

    max_freedom = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                max_freedom = max(max_freedom, bfs(r, c))
    
    print(max_freedom)

solve()