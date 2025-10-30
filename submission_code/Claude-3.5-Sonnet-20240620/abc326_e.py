# YOUR CODE HERE
MOD = 998244353

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def main():
    N = int(input())
    A = list(map(int, input().split()))

    dp = [0] * (N + 1)
    sum_dp = [0] * (N + 1)

    for i in range(N - 1, -1, -1):
        numerator = (N - i) * A[i] + sum_dp[i + 1]
        denominator = N - i
        dp[i] = (numerator * mod_inverse(denominator)) % MOD
        sum_dp[i] = (sum_dp[i + 1] + dp[i]) % MOD

    print(dp[0])

if __name__ == "__main__":
    main()