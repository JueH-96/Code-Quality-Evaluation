import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    input = sys.stdin.readline
    mod = 998244353

    N, M = map(int, input().split())
    # For each position p, we build maxL_p[r] = max L among constraints with X=p and R<=r
    maxL = [[0] * (N+1) for _ in range(N+1)]
    for _ in range(M):
        L, R, X = map(int, input().split())
        # constraint says: in segment covering [L,R], pivot at X is forbidden
        # record for X at R the L
        if maxL[X][R] < L:
            maxL[X][R] = L
    # prefix max by R
    for p in range(1, N+1):
        for r in range(1, N+1):
            if maxL[p][r-1] > maxL[p][r]:
                maxL[p][r] = maxL[p][r-1]

    # dp[i][j]: number of perms of positions i..j satisfying constraints
    # use 1-based; we only store i<=j
    dp = [[0]*(N+2) for _ in range(N+2)]
    # base: empty interval =1
    for i in range(1, N+2):
        dp[i][i-1] = 1

    # length from 1 to N
    for length in range(1, N+1):
        for i in range(1, N-length+2):
            j = i + length - 1
            res = 0
            # choose pivot p in [i..j] for the largest value
            # p is allowed if maxL[p][j] < i
            for p in range(i, j+1):
                if maxL[p][j] < i:
                    res += dp[i][p-1] * dp[p+1][j]
                    if res >= 1<<63:
                        res %= mod
            dp[i][j] = res % mod
    print(dp[1][N] % mod)

if __name__ == "__main__":
    main()