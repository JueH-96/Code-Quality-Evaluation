def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]

    while True:
        marked = [[False] * w for _ in range(h)]
        removed = False

        # Mark rows
        for r in range(h):
            if all(grid[r][c] != '.' for c in range(w)) and len(set(grid[r][c] for c in range(w) if grid[r][c] != '.')) == 1 and sum(1 for c in range(w) if grid[r][c] != '.') >= 2:
                for c in range(w):
                    if grid[r][c] != '.':
                        marked[r][c] = True
                        removed = True

        # Mark columns
        for c in range(w):
            col = [grid[r][c] for r in range(h)]
            if all(col[r] != '.' for r in range(h)) and len(set(col[r] for r in range(h) if col[r] != '.')) == 1 and sum(1 for r in range(h) if col[r] != '.') >= 2:
                for r in range(h):
                    if grid[r][c] != '.':
                        marked[r][c] = True
                        removed = True

        if not removed:
            break

        # Remove marked cookies
        for r in range(h):
            for c in range(w):
                if marked[r][c]:
                    grid[r][c] = '.'

    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != '.':
                count += 1
    print(count)

solve()