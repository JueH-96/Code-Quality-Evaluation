import sys
from itertools import combinations

readline = sys.stdin.readline

def solve():
    N = int(readline())
    D = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        d = list(map(int, readline().split()))
        for j, w in enumerate(d):
            D[i][i + j + 1] = D[i + j + 1][i] = w

    ans = 0
    for comb in combinations(range(N), 4):
        a, b, c, d = comb
        ans = max(ans, D[a][b] + D[c][d])
        ans = max(ans, D[a][c] + D[b][d])
        ans = max(ans, D[a][d] + D[b][c])
    for comb in combinations(range(N), 6):
        a, b, c, d, e, f = comb
        ans = max(ans, D[a][b] + D[c][d] + D[e][f])
        ans = max(ans, D[a][b] + D[c][e] + D[d][f])
        ans = max(ans, D[a][b] + D[c][f] + D[d][e])
        ans = max(ans, D[a][c] + D[b][d] + D[e][f])
        ans = max(ans, D[a][c] + D[b][e] + D[d][f])
        ans = max(ans, D[a][c] + D[b][f] + D[d][e])
        ans = max(ans, D[a][d] + D[b][c] + D[e][f])
        ans = max(ans, D[a][d] + D[b][e] + D[c][f])
        ans = max(ans, D[a][d] + D[b][f] + D[c][e])
        ans = max(ans, D[a][e] + D[b][c] + D[d][f])
        ans = max(ans, D[a][e] + D[b][d] + D[c][f])
        ans = max(ans, D[a][e] + D[b][f] + D[c][d])
        ans = max(ans, D[a][f] + D[b][c] + D[d][e])
        ans = max(ans, D[a][f] + D[b][d] + D[c][e])
        ans = max(ans, D[a][f] + D[b][e] + D[c][d])
    print(ans)

solve()