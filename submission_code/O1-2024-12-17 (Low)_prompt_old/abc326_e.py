def solve():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353

    # Compute inverse of N modulo MOD using Fermat's Little Theorem:
    invN = pow(N, MOD - 2, MOD)

    # Build suffix sum array S_A where:
    # S_A[i] = A[i] + A[i+1] + ... + A[N-1] (mod MOD)
    S_A = [0] * (N + 2)
    for i in range(N - 1, -1, -1):
        S_A[i] = (S_A[i + 1] + A[i]) % MOD

    # We'll compute dp[k], the expected amount earned starting from x = k.
    dp = [0] * (N + 1)
    # Also keep a suffix sum of dp: S_dp[k] = dp[k] + dp[k+1] + ... + dp[N] (mod MOD)
    S_dp = [0] * (N + 2)

    # Base case: dp[N] = 0 because once x = N, any roll is <= N, so we stop immediately.
    # Hence, S_dp[N] = dp[N] = 0, S_dp[N+1] = 0 as well by initialization.

    # Fill dp[] from k = N-1 down to k = 0
    for k in range(N - 1, -1, -1):
        # dp[k] = (1/N) * ( sum_{r = k+1..N} [A_r + dp[r]] )
        # which is (1/N) * ( S_A[k+1] + S_dp[k+1] )
        dp[k] = (S_A[k + 1] + S_dp[k + 1]) % MOD
        dp[k] = (dp[k] * invN) % MOD

        # Update the suffix sum for dp
        S_dp[k] = (dp[k] + S_dp[k + 1]) % MOD

    # dp[0] is the expected salary starting with x = 0
    print(dp[0] % MOD)

# Call solve() to execute when this file is run
if __name__ == "__main__":
    solve()