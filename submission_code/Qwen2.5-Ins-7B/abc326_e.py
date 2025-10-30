# N = int(input())
# A = list(map(int, input().split()))

def solve(N, A):
    mod = 998244353
    dp = [0] * (N + 1)
    dp[0] = 1
    for i in range(1, N + 1):
        for j in range(i, 0, -1):
            dp[j] = (dp[j] + dp[j - 1]) % mod
    ans = 0
    for i in range(N):
        ans = (ans + A[i] * dp[i + 1]) % mod
    return ans

N = int(input())
A = list(map(int, input().split()))
print(solve(N, A))