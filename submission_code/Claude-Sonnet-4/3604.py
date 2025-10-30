class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute factorials and their modular inverses for combinations
        def mod_inverse(a, mod):
            return pow(a, mod - 2, mod)
        
        def combination(n, r):
            if r > n or r < 0:
                return 0
            if r == 0 or r == n:
                return 1
            
            num = 1
            den = 1
            for i in range(r):
                num = (num * (n - i)) % MOD
                den = (den * (i + 1)) % MOD
            
            return (num * mod_inverse(den, MOD)) % MOD
        
        # Function to calculate number of surjective functions from n elements to k elements
        def surjective(n, k):
            if k > n:
                return 0
            if k == 0:
                return 1 if n == 0 else 0
            
            result = 0
            for i in range(k + 1):
                sign = 1 if i % 2 == 0 else -1
                term = combination(k, i) * pow(k - i, n, MOD)
                result = (result + sign * term) % MOD
            
            return result
        
        total = 0
        
        # For each possible number of non-empty stages
        for k in range(1, min(n, x) + 1):
            # Choose k stages out of x
            ways_choose_stages = combination(x, k)
            
            # Assign n performers to exactly k stages (all non-empty)
            ways_assign = surjective(n, k)
            
            # Award scores to k bands
            ways_score = pow(y, k, MOD)
            
            contribution = (ways_choose_stages * ways_assign % MOD * ways_score) % MOD
            total = (total + contribution) % MOD
        
        return total