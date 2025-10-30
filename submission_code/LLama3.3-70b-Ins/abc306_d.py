import sys

def solve():
    N = int(input())
    courses = []
    for _ in range(N):
        X, Y = map(int, input().split())
        courses.append((X, Y))

    dp = [[-float('inf')] * 2 for _ in range(N + 1)]
    dp[0][0] = 0

    for i in range(N):
        X, Y = courses[i]
        for j in range(2):
            # skip
            dp[i + 1][j] = max(dp[i + 1][j], dp[i][j])

            # eat
            if X == 0:  # antidotal
                if j == 0:  # healthy
                    dp[i + 1][0] = max(dp[i + 1][0], dp[i][0] + Y)
                else:  # upset
                    dp[i + 1][0] = max(dp[i + 1][0], dp[i][1] + Y)
            else:  # poisonous
                if j == 0:  # healthy
                    dp[i + 1][1] = max(dp[i + 1][1], dp[i][0] + Y)
                else:  # upset
                    continue

    print(max(dp[N][0], 0))

if __name__ == '__main__':
    solve()