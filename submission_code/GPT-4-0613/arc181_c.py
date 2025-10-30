N = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

P = [p-1 for p in P]
Q = [q-1 for q in Q]

P_inv = [0]*N
Q_inv = [0]*N

for i in range(N):
    P_inv[P[i]] = i
    Q_inv[Q[i]] = i

grid = [[0]*N for _ in range(N)]

for i in range(N):
    grid[P_inv[i]][Q_inv[i]] = 1

for row in grid:
    print(''.join(map(str, row)))