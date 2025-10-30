import math

MOD = 998244353

def main():
    N = int(input())
    A = list(map(int, input().split()))

    # Initialize the dp array to store the sum of scores for each length
    dp = [0] * (N + 1)

    # Initialize the prefix sum array to store the sum of gcd values
    prefix_sum = [0] * (N + 1)

    for i in range(1, N + 1):
        for j in range(i):
            prefix_sum[i] += math.gcd(A[j], A[i - 1])
            prefix_sum[i] %= MOD

    for i in range(1, N + 1):
        dp[i] = (dp[i - 1] + prefix_sum[i]) % MOD

    for i in range(1, N + 1):
        print(dp[i])

if __name__ == "__main__":
    main()