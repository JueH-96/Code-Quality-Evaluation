def solve():
    n = int(input())
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))

    grid = [['0' for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if p[i] < q[j]:
                grid[i][j] = '1'
            else:
                grid[i][j] = '0'

    for row in grid:
        print("".join(row))

solve()