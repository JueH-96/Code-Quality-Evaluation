class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7

        # Step 1: Compute Stirling numbers of the second kind S(n, k)
        # using dynamic programming
        S_prev = [0] * (x + 1)
        S_prev[0] = 1  # S(0,0) = 1
        for i in range(1, n + 1):
            S_curr = [0] * (x + 1)
            for j in range(1, min(i, x) + 1):
                S_curr[j] = (j * S_prev[j] + S_prev[j - 1]) % MOD
            S_prev = S_curr
        # After the loop, S_prev[k] contains S(n, k)

        # Step 2: Compute factorial and inverse factorial up to x
        fact = [1] * (x + 1)
        for i in range(1, x + 1):
            fact[i] = fact[i - 1] * i % MOD

        inv_fact = [1] * (x + 1)
        inv_fact[x] = pow(fact[x], MOD - 2, MOD)
        for i in range(x - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

        # Step 3: Calculate the result by summing over all possible k
        result = 0
        max_k = min(x, n)
        for k in range(1, max_k + 1):
            # Compute C(x, k) = fact[x] * inv_fact[k] * inv_fact[x - k] % MOD
            comb = fact[x] * inv_fact[k] % MOD
            comb = comb * inv_fact[x - k] % MOD

            # Compute term = C(x, k) * fact[k] * S(n, k) * y^k % MOD
            term = comb * fact[k] % MOD
            term = term * S_prev[k] % MOD
            term = term * pow(y, k, MOD) % MOD

            # Add to the result
            result = (result + term) % MOD

        return result