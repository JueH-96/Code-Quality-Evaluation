class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        # We need to compute C(k + n - 1, n - 1) mod MOD.
        # Precompute factorials and inverse factorials up to k + n.
        m = k + n
        fac = [1] * (m + 1)
        for i in range(1, m + 1):
            fac[i] = fac[i - 1] * i % MOD
        
        invfac = [1] * (m + 1)
        # Fermat's little theorem for modular inverse
        invfac[m] = pow(fac[m], MOD - 2, MOD)
        for i in range(m, 0, -1):
            invfac[i - 1] = invfac[i] * i % MOD
        
        # nCk helper
        def comb(x, y):
            if y < 0 or y > x:
                return 0
            return fac[x] * invfac[y] % MOD * invfac[x - y] % MOD
        
        # answer is C(k + (n-1), n-1)
        return comb(k + n - 1, n - 1)