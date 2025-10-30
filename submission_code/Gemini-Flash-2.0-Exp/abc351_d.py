def solve():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]

    def is_safe(r, c):
        if r < 0 or r >= H or c < 0 or c >= W:
            return False
        if grid[r][c] == '#':
            return False
        
        # Check adjacent cells for magnets
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '#':
                return False
        return True

    def bfs(start_r, start_c):
        if not is_safe(start_r, start_c):
            return 0

        q = [(start_r, start_c)]
        visited = set()
        visited.add((start_r, start_c))
        count = 1

        while q:
            r, c = q.pop(0)

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and is_safe(nr, nc):
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