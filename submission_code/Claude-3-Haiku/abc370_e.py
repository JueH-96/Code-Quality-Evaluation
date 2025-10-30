import sys

def count_valid_divisions(N, K, A):
    mod = 998244353
    dp = [0] * (2 ** N)
    dp[0] = 1

    for i in range(N):
        for j in range(2 ** i, 2 ** (i + 1)):
            dp[j] = (dp[j] + dp[j - (2 ** i)]) % mod
            if sum(A[i:j + 1]) == K:
                dp[j] = 0

    return sum(dp) % mod

N, K = map(int, input().split())
A = list(map(int, input().split()))
print(count_valid_divisions(N, K, A))