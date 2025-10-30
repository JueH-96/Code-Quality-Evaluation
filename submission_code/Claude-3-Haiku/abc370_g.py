from math import gcd

def is_good_integer(n):
    divisors = set()
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sum(divisors) % 3 == 0

def count_good_sequences(N, M):
    mod = 998244353
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    dp[1][1] = 1
    for i in range(2, N + 1):
        if is_good_integer(i):
            dp[i][1] = (dp[i - 1][1] + 1) % mod
        else:
            dp[i][1] = dp[i - 1][1]
        for j in range(2, M + 1):
            dp[i][j] = (dp[i - 1][j] + dp[i - 1][j - 1]) % mod
    return dp[N][M]

N, M = map(int, input().split())
print(count_good_sequences(N, M))