def solve():
    h, w, k = map(int, input().split())
    grid = [input() for _ in range(h)]

    min_ops = float('inf')

    # Check horizontal sequences
    for r in range(h):
        for c in range(w - k + 1):
            ops = 0
            for i in range(k):
                if grid[r][c + i] == '.':
                    ops += 1
            min_ops = min(min_ops, ops)

    # Check vertical sequences
    for c in range(w):
        for r in range(h - k + 1):
            ops = 0
            for i in range(k):
                if grid[r + i][c] == '.':
                    ops += 1
            min_ops = min(min_ops, ops)

    if min_ops == float('inf'):
        print(-1)
    else:
        print(min_ops)

solve()