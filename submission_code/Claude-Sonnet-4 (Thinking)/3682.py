class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Special case: m = 1
        if m == 1:
            return 1 if k == n - 1 else 0
        
        # General case: m >= 2
        # Formula: C(n-1, k) * m * (m-1)^(n-k-1)
        
        def mod_comb(n, k):
            if k > n or k < 0:
                return 0
            if k == 0 or k == n:
                return 1
            
            # Use the identity C(n, k) = C(n, n-k) to minimize iterations
            k = min(k, n - k)
            
            num = 1
            den = 1
            for i in range(k):
                num = (num * (n - i)) % MOD
                den = (den * (i + 1)) % MOD
            
            den_inv = pow(den, MOD - 2, MOD)
            return (num * den_inv) % MOD
        
        comb_val = mod_comb(n - 1, k)
        power_val = pow(m - 1, n - k - 1, MOD)
        
        result = (comb_val * m % MOD * power_val) % MOD
        return result