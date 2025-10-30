def solve():
    n = int(input())
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))

    p_inv = [0] * (n + 1)
    for i in range(n):
        p_inv[p[i]] = i

    q_inv = [0] * (n + 1)
    for i in range(n):
        q_inv[q[i]] = i

    grid = [['0'] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if p_inv[i + 1] < p_inv[j + 1] and q_inv[i + 1] > q_inv[j + 1]:
                grid[i][j] = '1'

    for row in grid:
        print("".join(row))

solve()