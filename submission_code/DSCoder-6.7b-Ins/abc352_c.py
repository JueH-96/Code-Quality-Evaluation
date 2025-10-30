import sys

def solve():
    N = int(sys.stdin.readline().strip())
    giants = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]

    giants.sort(key=lambda x: x[1] - x[0])

    dp = [0] * N
    dp[0] = giants[0][1]

    for i in range(1, N):
        dp[i] = max(dp[i-1] + giants[i][1], dp[i-1] + giants[i][0] + giants[i][1])

    print(dp[-1])

solve()