# YOUR CODE HERE
MOD = 998244353

def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))

    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + A[i]

    dp = [0] * (N + 1)
    dp[0] = 1

    stack = []
    for i in range(1, N + 1):
        while stack and prefix_sum[i] - prefix_sum[stack[-1]] == K:
            stack.pop()

        if stack:
            dp[i] = (dp[i - 1] * 2 - dp[stack[-1] - 1] + MOD) % MOD
        else:
            dp[i] = dp[i - 1] * 2 % MOD

        stack.append(i)

    print(dp[N])

solve()