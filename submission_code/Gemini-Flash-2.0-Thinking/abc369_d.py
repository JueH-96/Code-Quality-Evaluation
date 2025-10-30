def solve():
    n = int(input())
    a = list(map(int, input().split()))

    dp = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for k in range(i + 1):
            # Option 1: Let the i-th monster go
            dp[i][k] = dp[i - 1][k]

            # Option 2: Defeat the i-th monster
            if k > 0:
                experience = a[i - 1]
                if k % 2 == 0:
                    experience *= 2
                dp[i][k] = max(dp[i][k], dp[i - 1][k - 1] + experience)

    print(max(dp[n]))

if __name__ == "__main__":
    solve()