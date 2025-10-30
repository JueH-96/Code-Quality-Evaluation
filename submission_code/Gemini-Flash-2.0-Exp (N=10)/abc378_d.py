def solve():
    h, w, k = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w and grid[r][c] == '.'

    def count_paths(r, c, moves_left, visited):
        if moves_left == 0:
            return 1
        
        count = 0
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and (nr, nc) not in visited:
                count += count_paths(nr, nc, moves_left - 1, visited | {(nr, nc)})
        return count

    total_paths = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                total_paths += count_paths(r, c, k, {(r, c)})
    print(total_paths)

solve()