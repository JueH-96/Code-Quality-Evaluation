def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    mod = 998244353

    q_count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '?':
                q_count += 1

    def is_valid(r, c, val, current_grid):
        if not (0 <= r < h and 0 <= c < w):
            return True
        if current_grid[r][c] == '?':
            return True
        return current_grid[r][c] != str(val)

    def count_valid_grids(current_grid):
        q_indices = []
        for r in range(h):
            for c in range(w):
                if current_grid[r][c] == '?':
                    q_indices.append((r, c))

        if not q_indices:
            for r in range(h):
                for c in range(w):
                    if r > 0 and current_grid[r][c] == current_grid[r-1][c]:
                        return 0
                    if c > 0 and current_grid[r][c] == current_grid[r][c-1]:
                        return 0
            return 1

        count = 0
        r, c = q_indices[0]
        for val in range(1, 4):
            if (is_valid(r - 1, c, val, current_grid) and
                is_valid(r + 1, c, val, current_grid) and
                is_valid(r, c - 1, val, current_grid) and
                is_valid(r, c + 1, val, current_grid)):
                
                new_grid = [row[:] for row in current_grid]
                new_grid[r][c] = str(val)
                count = (count + count_valid_grids(new_grid)) % mod
        return count

    print(count_valid_grids(grid))

solve()