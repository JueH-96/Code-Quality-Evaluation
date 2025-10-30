class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        mod = 10**9 + 7
        
        def nCr(n_val, r_val, mod_val):
            if r_val < 0 or r_val > n_val:
                return 0
            if r_val == 0:
                return 1
            r_min = min(r_val, n_val - r_val)
            inv = [0] * (r_min + 1)
            inv[1] = 1
            for i in range(2, r_min + 1):
                inv[i] = (mod_val - (mod_val // i) * inv[mod_val % i] % mod_val) % mod_val
            res = 1
            for i in range(1, r_min + 1):
                res = (res * (n_val - i + 1)) % mod_val
                res = (res * inv[i]) % mod_val
            return res
        
        binom_val = nCr(n - 1, k, mod)
        exp_val = pow(m - 1, n - k - 1, mod)
        result = binom_val * m % mod * exp_val % mod
        return result