def solve(N, P, Q):
    grid = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if P[i] > P[j] and Q[i] < Q[j]:
                grid[i][j] = 1
    return grid

N = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

grid = solve(N, P, Q)
for row in grid:
    print(''.join(map(str, row)))