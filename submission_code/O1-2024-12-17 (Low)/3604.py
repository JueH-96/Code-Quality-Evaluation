class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute factorials and inverse factorials up to 2*n or just up to x+n, 
        # but since n,x <= 1000, up to 2000 is safe.
        # However, we only need up to 2000 for safety (or 1000 + 1000).
        MAX_VAL = 2000
        
        fact = [1] * (MAX_VAL+1)
        inv_fact = [1] * (MAX_VAL+1)
        
        # Compute factorials modulo
        for i in range(2, MAX_VAL+1):
            fact[i] = fact[i-1] * i % MOD
        
        # Fermat's little theorem for inverse factorial: inv_fact[n] = fact[n]^(MOD-2) mod MOD
        inv_fact[MAX_VAL] = pow(fact[MAX_VAL], MOD-2, MOD)
        for i in range(MAX_VAL, 0, -1):
            inv_fact[i-1] = inv_fact[i] * i % MOD
        
        def comb(n_, r_):
            if r_ < 0 or r_ > n_:
                return 0
            return fact[n_] * inv_fact[r_] % MOD * inv_fact[n_-r_] % MOD
        
        # onto(k, n) = number of ways to assign n distinct performers onto k distinct stages
        # such that each stage has at least one performer.
        # onto(k, n) = sum_{i=0..k} [(-1)^i * C(k, i) * (k-i)^n].
        def onto(k_, n_):
            total = 0
            for i in range(k_+1):
                # term = (-1)^i * C(k_, i) * (k_-i)^n_
                sign = -1 if i % 2 else 1
                ways = comb(k_, i) * pow(k_ - i, n_, MOD) % MOD
                if sign == 1:
                    total = (total + ways) % MOD
                else:
                    total = (total - ways) % MOD
            return total % MOD
        
        # Sum over the number of used stages k = 1..min(x,n)
        # First choose which k stages out of x: comb(x, k)
        # Then onto(k,n) ways to assign n performers to those k stages (all used)
        # Then y^k ways to assign scores to those k bands
        result = 0
        for k in range(1, x+1):
            if k > n: 
                break
            ways_assign = comb(x, k) * onto(k, n) % MOD
            ways_score = pow(y, k, MOD)
            result = (result + ways_assign * ways_score) % MOD
        
        return result