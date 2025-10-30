def solve():
    H, W, K = map(int, input().split())
    grid = [input() for _ in range(H)]

    ans = float('inf')

    for i in range(H):
        for j in range(W - K + 1):
            cost = 0
            possible = True
            for k in range(K):
                if grid[i][j+k] == 'x':
                    possible = False
                    break
                elif grid[i][j+k] == '.':
                    cost += 1
            if possible:
                ans = min(ans, cost)

    for j in range(W):
        for i in range(H - K + 1):
            cost = 0
            possible = True
            for k in range(K):
                if grid[i+k][j] == 'x':
                    possible = False
                    break
                elif grid[i+k][j] == '.':
                    cost += 1
            if possible:
                ans = min(ans, cost)

    if ans == float('inf'):
        print(-1)
    else:
        print(ans)

solve()