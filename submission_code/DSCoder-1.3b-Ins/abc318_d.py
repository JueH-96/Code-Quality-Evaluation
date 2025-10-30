import sys

def solve():
    N = int(sys.stdin.readline().strip())
    weights = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]
    weights.append([0]*N)
    for i in range(N):
        for j in range(i+1, N):
            if weights[i][j] == 0:
                weights[i][j] = weights[j][i]
    dp = [0]*(1 << N)
    for mask in range(1, 1 << N):
        for i in range(N):
            if ((mask >> i) & 1) == 0:
                continue
            nxt_mask = mask ^ (1 << i)
            for j in range(i):
                if ((nxt_mask >> j) & 1) == 0:
                    continue
                dp[nxt_mask] = max(dp[nxt_mask], dp[mask] + weights[i][j])
    print(dp[(1 << N) - 1])

solve()