import sys

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    # dp[i][0]: number of ways to divide A[0:i] such that the last subsequence sums to K
    # dp[i][1]: number of ways to divide A[0:i] such that the last subsequence does not sum to K
    dp = [[0, 1] for _ in range(N + 1)]

    current_sum = 0
    sum_count = {0: 1}

    for i in range(1, N + 1):
        current_sum = (current_sum + A[i-1]) % MOD

        # Count the number of ways to get a sum of `current_sum - K`
        dp[i][0] = sum_count.get(current_sum - K, 0)
        dp[i][1] = (2 * dp[i-1][1] + dp[i-1][0] - dp[i][0]) % MOD

        # Update the sum count
        sum_count[current_sum] = sum_count.get(current_sum, 0) + 1

    print(dp[N][1])

if __name__ == "__main__":
    main()