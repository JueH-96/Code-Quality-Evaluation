# YOUR CODE HERE
def solve():
    h, w, y = map(int, input().split())
    a = []
    for _ in range(h):
        a.append(list(map(int, input().split())))

    def get_area(level):
        grid = [row[:] for row in a]
        q = []
        
        for r in range(h):
            for c in range(w):
                if grid[r][c] <= level:
                    if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                        q.append((r, c))

        while q:
            r, c = q.pop(0)
            if 0 <= r < h and 0 <= c < w and grid[r][c] <= level:
                grid[r][c] = -1 
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] <= level and grid[nr][nc] != -1:
                        q.append((nr, nc))

        area = 0
        for r in range(h):
            for c in range(w):
                if grid[r][c] > level:
                    area += 1
        return area

    for i in range(1, y + 1):
        print(get_area(i))

solve()