def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    mod = 998244353

    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        for j in range(i):
            subsequence_sum = sum(a[j:i])
            if subsequence_sum != k:
                dp[i] = (dp[i] + dp[j]) % mod

    print(dp[n])

if __name__ == "__main__":
    solve()