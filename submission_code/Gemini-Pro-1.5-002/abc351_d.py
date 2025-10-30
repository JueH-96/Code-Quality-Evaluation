# YOUR CODE HERE
def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w

    def has_adjacent_magnet(r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if abs(dr) + abs(dc) != 1:
                    continue
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc) and grid[nr][nc] == '#':
                    return True
        return False

    def bfs(start_r, start_c):
        q = [(start_r, start_c)]
        visited = set([(start_r, start_c)])
        count = 0
        
        while q:
            r, c = q.pop(0)
            count += 1
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if abs(dr) + abs(dc) != 1:
                        continue
                    nr, nc = r + dr, c + dc
                    if (
                        is_valid(nr, nc) and
                        grid[nr][nc] == '.' and
                        (nr, nc) not in visited and
                        not has_adjacent_magnet(nr, nc)
                    ):
                        q.append((nr, nc))
                        visited.add((nr, nc))
        return count

    max_freedom = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.' and not has_adjacent_magnet(r, c):
                max_freedom = max(max_freedom, bfs(r, c))

    print(max_freedom)

solve()