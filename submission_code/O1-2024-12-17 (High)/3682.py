class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Special case when n = 1
        # There's only one element, so k must be 0 to have no adjacent equal pairs.
        if n == 1:
            return m % MOD if k == 0 else 0
        
        # Precompute factorials and inverse factorials up to n (which is enough for n-1 as well)
        fact = [1] * (n + 1)
        invfact = [1] * (n + 1)
        
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i % MOD
        
        # Fermat's little theorem for inverse factorial
        invfact[n] = pow(fact[n], MOD - 2, MOD)  # inverse of fact[n]
        for i in range(n - 1, -1, -1):
            invfact[i] = invfact[i + 1] * (i + 1) % MOD
        
        # nCr function under modulo
        def nCr(n_, r_):
            if r_ < 0 or r_ > n_:
                return 0
            return fact[n_] * invfact[r_] % MOD * invfact[n_ - r_] % MOD
        
        # Formula:
        # count = m * C(n-1, k) * (m-1)^(n-1-k)
        choose = nCr(n - 1, k)
        diff_ways = pow(m - 1, n - 1 - k, MOD)
        ans = (m % MOD) * choose % MOD
        ans = ans * diff_ways % MOD
        
        return ans