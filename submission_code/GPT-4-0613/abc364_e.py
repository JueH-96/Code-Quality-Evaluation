import sys
from operator import itemgetter

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    AB = sorted([tuple(map(int, sys.stdin.readline().split())) for _ in range(N)], key=itemgetter(0)+itemgetter(1), reverse=True)
    dp = [[[-1]*(Y+1) for _ in range(X+1)] for _ in range(N+1)]
    dp[0][0][0] = 0
    for i in range(N):
        a, b = AB[i]
        for j in range(X+1):
            for k in range(Y+1):
                if dp[i][j][k] == -1:
                    continue
                dp[i+1][min(j+a, X)][min(k+b, Y)] = max(dp[i+1][min(j+a, X)][min(k+b, Y)], dp[i][j][k]+1)
                dp[i+1][j][k] = max(dp[i+1][j][k], dp[i][j][k])
    print(max(dp[N][X]))

if __name__ == "__main__":
    solve()