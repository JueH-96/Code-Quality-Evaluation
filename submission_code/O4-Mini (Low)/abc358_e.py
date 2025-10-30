def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    K = int(input().strip())
    C = list(map(int, input().split()))
    # dp[n]: number of ways to pick counts summing to n after processing some letters
    dp = [0] * (K + 1)
    dp[0] = 1

    # Process each letter with its cap Ci via prefix sums (sliding window convolution)
    for Ci in C:
        # build prefix sums of old dp
        prefix = [0] * (K + 1)
        s = 0
        for i in range(K + 1):
            s = (s + dp[i]) % MOD
            prefix[i] = s
        new_dp = [0] * (K + 1)
        for n in range(K + 1):
            # sum_{k=0..Ci} dp[n-k] = prefix[n] - prefix[n-Ci-1] if that index >=0
            lo = n - Ci - 1
            if lo >= 0:
                val = prefix[n] - prefix[lo]
            else:
                val = prefix[n]
            new_dp[n] = val % MOD
        dp = new_dp

    # Precompute factorials up to K
    fac = [1] * (K + 1)
    for i in range(1, K + 1):
        fac[i] = fac[i-1] * i % MOD

    # Sum over lengths 1..K of dp[n] * n!
    ans = 0
    for n in range(1, K + 1):
        ans = (ans + dp[n] * fac[n]) % MOD

    print(ans)


if __name__ == "__main__":
    main()