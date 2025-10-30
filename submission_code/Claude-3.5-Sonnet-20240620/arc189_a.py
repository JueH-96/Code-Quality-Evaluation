# YOUR CODE HERE
MOD = 998244353

def solve(N, A):
    dp = [0] * (N + 1)
    last = [-1, -1]
    dp[0] = 1

    for i in range(N):
        dp[i + 1] = dp[i]
        if last[A[i]] != -1:
            l = last[1 - A[i]]
            if l < last[A[i]]:
                dp[i + 1] = (dp[i + 1] + dp[l + 1]) % MOD
        last[A[i]] = i

    return dp[N]

N = int(input())
A = list(map(int, input().split()))

result = solve(N, A)
print(result)