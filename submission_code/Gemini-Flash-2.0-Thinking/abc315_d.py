def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]

    while True:
        marked = False
        marks = [[False for _ in range(w)] for _ in range(h)]

        # Mark rows
        for r in range(h):
            row_cookies = []
            row_indices = []
            for c in range(w):
                if grid[r][c] != '.':
                    row_cookies.append(grid[r][c])
                    row_indices.append(c)

            if len(row_cookies) >= 2 and all(cookie == row_cookies[0] for cookie in row_cookies):
                for c_idx in row_indices:
                    marks[r][c_idx] = True
                    marked = True

        # Mark columns
        for c in range(w):
            col_cookies = []
            col_indices = []
            for r in range(h):
                if grid[r][c] != '.':
                    col_cookies.append(grid[r][c])
                    col_indices.append(r)

            if len(col_cookies) >= 2 and all(cookie == col_cookies[0] for cookie in col_cookies):
                for r_idx in col_indices:
                    marks[r_idx][c] = True
                    marked = True

        # Remove marked cookies
        if not marked:
            break

        for r in range(h):
            for c in range(w):
                if marks[r][c]:
                    grid[r][c] = '.'

    remaining_cookies = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != '.':
                remaining_cookies += 1

    print(remaining_cookies)

solve()