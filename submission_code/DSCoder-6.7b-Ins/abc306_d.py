import sys

def solve():
    N = int(sys.stdin.readline().strip())
    courses = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]

    dp = [[0, 0] for _ in range(N+1)]
    for i in range(N-1, -1, -1):
        if courses[i][0] == 0:
            dp[i][0] = max(dp[i+1][0], dp[i+1][1]+courses[i][1])
            dp[i][1] = max(dp[i+1][1], dp[i+1][0])
        else:
            dp[i][0] = max(dp[i+1][0], dp[i+1][1])
            dp[i][1] = max(dp[i+1][1]+courses[i][1], dp[i+1][0])

    print(max(dp[0]))

solve()