class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        max_fact = 1000  # since x and n can be up to 1000

        # Precompute factorial and inverse factorial
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD

        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
        for i in range(max_fact - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

        # Function to compute combination C(a, b)
        def comb(a, b):
            if b < 0 or b > a:
                return 0
            return fact[a] * inv_fact[b] % MOD * inv_fact[a - b] % MOD

        # Precompute C(x choose k) for all k in 0..x
        C_x = [comb(x, k) for k in range(x + 1)]

        # Precompute y_pows up to x
        y_pows = [1] * (x + 1)
        for k in range(1, x + 1):
            y_pows[k] = y_pows[k - 1] * y % MOD

        # Precompute surj(n, k) for k in 0..x
        surj = [0] * (x + 1)
        for k in range(x + 1):
            if k == 0:
                surj[k] = 0
                continue
            surj_val = 0
            for i in range(k + 1):
                c = comb(k, i)
                sign = 1 if (i % 2 == 0) else MOD - 1
                term = c * sign % MOD
                pow_term = pow(k - i, n, MOD)
                term = term * pow_term % MOD
                surj_val = (surj_val + term) % MOD
            surj[k] = surj_val

        # Calculate the total sum
        total = 0
        for k in range(x + 1):
            term = C_x[k] * y_pows[k] % MOD
            term = term * surj[k] % MOD
            total = (total + term) % MOD

        return total