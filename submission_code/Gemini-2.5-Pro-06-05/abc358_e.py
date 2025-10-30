import sys

def solve():
    """
    Solves the AtCoder nameplate problem using dynamic programming.
    """
    MOD = 998244353

    # Read input from stdin
    try:
        K = int(sys.stdin.readline())
        C = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    # Precompute factorials and their modular inverses for combinations C(n, k)
    # The maximum 'n' in C(n, k) is K, as the string length is at most K.
    MAX_N = K
    fact = [1] * (MAX_N + 1)
    invfact = [1] * (MAX_N + 1)

    for i in range(1, MAX_N + 1):
        fact[i] = (fact[i - 1] * i) % MOD

    invfact[MAX_N] = pow(fact[MAX_N], MOD - 2, MOD)
    for i in range(MAX_N, 0, -1):
        invfact[i - 1] = (invfact[i] * i) % MOD

    def combinations(n, k):
        """Calculates C(n, k) % MOD using precomputed values."""
        if k < 0 or k > n:
            return 0
        numerator = fact[n]
        denominator = (invfact[k] * invfact[n - k]) % MOD
        return (numerator * denominator) % MOD

    # DP state: dp[j] = number of ways to form a string of length j
    # using the character types considered so far.
    dp = [0] * (K + 1)
    dp[0] = 1  # Base case: one way to form an empty string.

    # Iterate over each of the 26 character types
    for c_limit in C:
        new_dp = [0] * (K + 1)
        # j: length of strings from previous character types
        for j in range(K + 1):
            if dp[j] == 0:
                continue
            # k: number of current character type to add
            for k in range(c_limit + 1):
                new_len = j + k
                if new_len > K:
                    break
                
                # Number of ways to place k new identical characters is C(new_len, k)
                comb = combinations(new_len, k)
                
                # Update the DP table for the new length
                term = (dp[j] * comb) % MOD
                new_dp[new_len] = (new_dp[new_len] + term) % MOD
        
        dp = new_dp

    # The result is the sum of ways for all lengths from 1 to K.
    result = sum(dp[1:]) % MOD

    print(result)

solve()