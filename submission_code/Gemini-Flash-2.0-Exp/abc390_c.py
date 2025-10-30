def solve():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]

    min_row, max_row = -1, -1
    min_col, max_col = -1, -1

    for r in range(H):
        for c in range(W):
            if grid[r][c] == '#':
                if min_row == -1:
                    min_row = r
                max_row = r
                if min_col == -1:
                    min_col = c
                max_col = c

    if min_row == -1:
        print("Yes")
        return

    for r in range(H):
        for c in range(W):
            if min_row <= r <= max_row and min_col <= c <= max_col:
                if grid[r][c] == '.':
                    print("No")
                    return
            else:
                if grid[r][c] == '#':
                    print("No")
                    return

    print("Yes")

solve()