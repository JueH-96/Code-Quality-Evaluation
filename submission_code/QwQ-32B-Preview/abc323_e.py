import sys
import math

def main():
    MOD = 998244353
    N, X = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))
    T1 = T[0]

    # Convert times to half-seconds
    X_half = 2 * X + 1
    T_half = [2 * t for t in T]

    # Define the DP array
    max_t = X_half
    dp = [0] * (max_t + 1)
    dp[0] = 1

    # Precompute the inverse of (N - 1)
    inv_N_minus_1 = pow(N - 1, MOD - 2, MOD)

    # Compute dp[t]
    for t in range(max_t + 1):
        for i in range(1, N):
            t_i = T_half[i]
            if t >= t_i:
                dp[t] = (dp[t] + dp[t - t_i] * inv_N_minus_1) % MOD

    # Calculate the lower and upper bounds for the valid t range
    lower = max(0, math.ceil(X_half - T_half[0]))
    upper = X_half - 1  # since the range is [lower, upper]

    # Sum dp[t] for t in [lower, upper]
    total = 0
    for t in range(lower, upper + 1):
        total = (total + dp[t]) % MOD

    # Multiply by the probability of choosing song 1 (1 / N)
    inv_N = pow(N, MOD - 2, MOD)
    result = (total * inv_N) % MOD

    print(result)

if __name__ == "__main__":
    main()