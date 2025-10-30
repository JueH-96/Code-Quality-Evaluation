def solve():
    h, w = map(int, input().split())
    s = [list(input()) for _ in range(h)]
    modulo = 998244353

    def is_safe(r, c, val, grid):
        if r > 0 and grid[r - 1][c] == val:
            return False
        if c > 0 and grid[r][c - 1] == val:
            return False
        return True

    def backtrack(r, c, current_grid):
        if r == h:
            return 1

        next_r, next_c = (r, c + 1) if c < w - 1 else (r + 1, 0)

        if s[r][c] != '?':
            return backtrack(next_r, next_c, current_grid)
        else:
            count = 0
            for val in ['1', '2', '3']:
                if is_safe(r, c, val, current_grid):
                    current_grid[r][c] = val
                    count = (count + backtrack(next_r, next_c, current_grid)) % modulo
                    current_grid[r][c] = '?'
            return count

    initial_grid = [list(row) for row in s]
    print(backtrack(0, 0, initial_grid))

solve()