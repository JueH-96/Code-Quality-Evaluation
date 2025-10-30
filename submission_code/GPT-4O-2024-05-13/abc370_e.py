# YOUR CODE HERE
import sys
input = sys.stdin.read
MOD = 998244353

def solve():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    # Precompute prefix sums
    prefix_sums = [0] * (N + 1)
    for i in range(N):
        prefix_sums[i + 1] = prefix_sums[i] + A[i]

    # dp[i] will be the number of valid ways to divide the first i elements
    dp = [0] * (N + 1)
    dp[0] = 1  # There's one way to divide an empty array

    # last_occurrence stores the last occurrence of each prefix sum
    last_occurrence = {}
    last_occurrence[0] = 0

    for i in range(1, N + 1):
        if prefix_sums[i] - K in last_occurrence:
            dp[i] = dp[last_occurrence[prefix_sums[i] - K]]
        dp[i] = (dp[i] + dp[i - 1]) % MOD
        last_occurrence[prefix_sums[i]] = i

    # The result is the number of ways to divide the entire array
    result = dp[N]
    print(result)

solve()