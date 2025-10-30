def solve():
    n, m = map(int, input().split())
    grid = [input() for _ in range(n)]
    result = []

    def is_tak(r, c):
        # Check top-left 3x3
        for i in range(r, r + 3):
            for j in range(c, c + 3):
                if grid[i][j] != '#':
                    return False

        # Check bottom-right 3x3
        for i in range(r + 6, r + 9):
            for j in range(c + 6, c + 9):
                if grid[i][j] != '#':
                    return False

        # Check adjacent cells to top-left 3x3
        for i in range(r - 1, r + 4):
            for j in range(c - 1, c + 4):
                if 0 <= i < n and 0 <= j < m:
                    if not (r <= i < r + 3 and c <= j < c + 3):
                        if grid[i][j] == '#':
                            return False

        # Check adjacent cells to bottom-right 3x3
        for i in range(r + 5, r + 10):
            for j in range(c + 5, c + 10):
                if 0 <= i < n and 0 <= j < m:
                    if not (r + 6 <= i < r + 9 and c + 6 <= j < c + 9):
                        if grid[i][j] == '#':
                            return False
        return True

    for r in range(n - 8):
        for c in range(m - 8):
            if is_tak(r, c):
                result.append((r + 1, c + 1))

    result.sort()
    for r, c in result:
        print(r, c)

solve()