import itertools
import sys
sys.setrecursionlimit(10**9)
read = sys.stdin.read
readline = sys.stdin.readline
readlines = sys.stdin.readlines

N = int(readline())
D = (list(map(int, line.split())) for line in read().splitlines())

dy = [0] * N
for mask in range(1, 1 << N):
    for i, j in itertools.combinations(range(N), 2):
        if mask & (1 << i) and mask & (1 << j):
            k, dy[i], dy[j] = 0, dy[i], dy[j]
            if mask == 3:
                dy[i], dy[j] = D[0][1], D[1][0]
            else:
                for x in range(N):
                    if x != i and x != j and mask & (1 << x):
                        dy[i], dy[j] = dy[i] + D[x][j], dy[j] + D[i][x]
                        k += 1
            if k:
                dy[i], dy[j] = dy[i] + D[i][j] // k, dy[j] + D[i][j] // k

dp = [dy]
for _ in range(N-1):
    ndp = [[] for _ in range(N)]
    for mask in range(1, 1 << N):
        for j in range(N):
            if mask & (1 << j):
                msk = mask ^ (1 << j)
                if not msk:
                    continue
                w1 = D[j][j ^ msk.bit_count() - 1 ^ N + msk]
                w2 = dp[-1][j][msk]
                ndp[j].append(max(w1, w2))
    dp.append(ndp)

print(max(dp[-1][i][-1] for i in range(N)))