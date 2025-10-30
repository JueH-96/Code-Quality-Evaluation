def solve():
    h, w, k = map(int, input().split())
    grid = [input() for _ in range(h)]
    ans = float('inf')

    for r in range(h):
        for c in range(w - k + 1):
            count = 0
            dots = 0
            for i in range(k):
                if grid[r][c + i] == 'x':
                    count = float('inf')
                    break
                elif grid[r][c + i] == '.':
                    dots += 1
            ans = min(ans, count + dots)

    for c in range(w):
        for r in range(h - k + 1):
            count = 0
            dots = 0
            for i in range(k):
                if grid[r + i][c] == 'x':
                    count = float('inf')
                    break
                elif grid[r + i][c] == '.':
                    dots += 1
            ans = min(ans, count + dots)

    if ans == float('inf'):
        print(-1)
    else:
        print(ans)

solve()