class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        mod = 10**9 + 7
        
        # If m == 1, the only array possible is all 1's,
        # which has exactly n-1 equal adjacent pairs.
        if m == 1:
            return 1 if k == n - 1 else 0
        
        # We use the formula:
        #   #good arrays = C(n-1, k) * m * (m-1)^( (n-1) - k )  (mod mod)
        # where C(n-1, k) is "n-1 choose k".
        
        # Precompute factorials and inverse factorials up to n-1
        fac = [1] * (n)
        for i in range(1, n):
            fac[i] = fac[i-1] * i % mod
        
        invfac = [1] * (n)
        invfac[n-1] = pow(fac[n-1], mod-2, mod)
        for i in range(n-1, 0, -1):
            invfac[i-1] = invfac[i] * i % mod
        
        # Compute C(n-1, k)
        if k > n-1:
            return 0
        comb = fac[n-1] * invfac[k] % mod * invfac[n-1-k] % mod
        
        # Compute the power (m-1)^(n-1-k)
        pow_part = pow(m-1, n-1-k, mod)
        
        # Final answer
        return comb * m % mod * pow_part % mod