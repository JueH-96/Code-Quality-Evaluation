def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    MOD = 998244353

    K = int(input_data[0])
    C = list(map(int, input_data[1:]))

    # Precompute factorials and inverse factorials up to K
    fact = [1] * (K+1)
    inv_fact = [1] * (K+1)
    for i in range(1, K+1):
        fact[i] = fact[i-1] * i % MOD

    # Fermat's little theorem for inverse factorial
    inv_fact[K] = pow(fact[K], MOD-2, MOD)
    for i in reversed(range(1, K)):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

    # dp[L] will hold the coefficient of x^L in the product of EGFs (sum_{x_i} 1/(x_1!...x_26!)),
    # i.e. how many ways to have a distribution summing to L.
    dp = [0] * (K+1)
    dp[0] = 1

    # Multiply polynomials (EGF factors) for each letter
    for i in range(26):
        limit = min(C[i], K)
        new_dp = [0] * (K+1)
        for length_used in range(K+1):
            if dp[length_used] != 0:
                base_val = dp[length_used]
                # Convolve with 1 + x/1! + x^2/2! + ... + x^(limit)/(limit!)
                for t in range(limit+1):
                    nxt = length_used + t
                    if nxt > K:
                        break
                    new_dp[nxt] = (new_dp[nxt] + base_val * inv_fact[t]) % MOD
        dp = new_dp

    # Now dp[L] is sum of 1/(x_1!... x_26!) for distributions summing to L.
    # The number of distinct strings of length L is dp[L] * L!.
    ans = 0
    for length in range(1, K+1):
        ans = (ans + dp[length] * fact[length]) % MOD

    print(ans)

# Call main() to execute the solution.
if __name__ == "__main__":
    main()