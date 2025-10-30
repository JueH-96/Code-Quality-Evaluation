import sys
input = sys.stdin.readline

MOD = 998244353

def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))

    # Calculate prefix sums
    prefix_sums = [0] * (N + 1)
    for i in range(N):
        prefix_sums[i + 1] = prefix_sums[i] + A[i]

    # Initialize dp array
    dp = [0] * (N + 1)
    dp[0] = 1

    # Iterate over all possible subsequences
    for i in range(1, N + 1):
        for j in range(i):
            # Check if the sum of the subsequence is not equal to K
            if prefix_sums[i] - prefix_sums[j] != K:
                dp[i] += dp[j]
                dp[i] %= MOD

    # The answer is the sum of all dp values
    ans = sum(dp) % MOD
    print(ans)

if __name__ == "__main__":
    solve()