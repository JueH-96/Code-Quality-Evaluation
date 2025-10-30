# YOUR CODE HERE
import sys
input = sys.stdin.readline

def solve():
    N, X, Y = map(int, input().split())
    P = []
    T = []
    for _ in range(N-1):
        p, t = map(int, input().split())
        P.append(p)
        T.append(t)
    Q = int(input())
    queries = list(map(int, input().split()))

    # Precompute the earliest arrival times for each bus stop
    dp = [[float('inf')] * 8 for _ in range(N)]
    dp[0][0] = X
    for i in range(N-1):
        for j in range(8):
            if dp[i][j] != float('inf'):
                # Walking to the next bus stop
                dp[i+1][0] = min(dp[i+1][0], dp[i][j] + T[i])
                # Taking the bus to the next bus stop
                for k in range(P[i]):
                    dp[i+1][(j+k) % P[i]] = min(dp[i+1][(j+k) % P[i]], dp[i][j] + k + T[i])

    # Answer each query
    for q in queries:
        min_time = float('inf')
        for j in range(8):
            if dp[N-1][j] != float('inf'):
                min_time = min(min_time, dp[N-1][j] + Y + ((P[N-2] - (q + X) % P[N-2]) % P[N-2] if (q + X) % P[N-2] != 0 else 0))
        print(min_time)

solve()