class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        if s == t:
            target_shift = 0
        else:
            target_shift = -1
            for shift in range(1, n):
                shifted_s = s[shift:] + s[:shift]
                if shifted_s == t:
                    target_shift = shift
                    break
            if target_shift == -1:
                return 0
                
        MOD = 10**9 + 7
        
        def power(a, b):
            res = 1
            a %= MOD
            while b > 0:
                if b % 2 == 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                b //= 2
            return res
            
        n_inv = power(n, MOD - 2)
        term1 = power(n - 1, k)
        term2 = power(-1, k)
        
        if target_shift == 0:
            result = (term1 + (term2 * (n - 1)) % MOD) % MOD
        else:
            result = (term1 - term2 + MOD) % MOD
            
        result = (result * n_inv) % MOD
        return result