import sys

def solve():
    N = int(sys.stdin.readline().strip())
    P = list(map(int, sys.stdin.readline().strip().split()))
    Q = list(map(int, sys.stdin.readline().strip().split()))

    grid = [['0' for _ in range(N)] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if P[i] < Q[j]:
                grid[i][j] = '0'
            else:
                grid[i][j] = '1'

    for row in grid:
        print(''.join(row))

solve()