def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    snuke = "snuke"
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w

    def dfs(r, c, index):
        if r == h - 1 and c == w - 1:
            return True
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and grid[nr][nc] == snuke[(index + 1) % 5]:
                if (nr, nc, (index + 1) % 5) not in visited:
                    visited.add((nr, nc, (index + 1) % 5))
                    if dfs(nr, nc, (index + 1) % 5):
                        return True
                    visited.remove((nr, nc, (index + 1) % 5))
        return False

    visited = set()
    visited.add((0, 0, 0))
    if grid[0][0] == 's' and dfs(0, 0, 0):
        print("Yes")
    else:
        print("No")

solve()