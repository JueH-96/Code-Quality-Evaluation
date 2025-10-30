def solve():
    k = int(input())
    c = list(map(int, input().split()))
    mod = 998244353

    dp = [[0] * (k + 1) for _ in range(27)]
    dp[0][0] = 1

    for i in range(1, 27):
        for j in range(k + 1):
            for count in range(c[i - 1] + 1):
                if j - count >= 0:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - count]) % mod

    ans = 0
    for length in range(1, k + 1):
        ans = (ans + dp[26][length]) % mod

    print(ans)

if __name__ == "__main__":
    solve()