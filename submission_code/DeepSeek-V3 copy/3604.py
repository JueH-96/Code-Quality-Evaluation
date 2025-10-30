MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        # Precompute the number of ways to partition n performers into x stages
        # This is equivalent to the number of ways to assign each performer to one of x stages
        # which is x^n
        stage_assignments = pow(x, n, MOD)
        
        # Precompute the number of ways to assign scores to the bands
        # The number of bands can range from 1 to x
        # For each possible number of bands k (1 <= k <= x), we need to:
        # 1. Choose k stages out of x to have at least one performer
        # 2. Assign y possible scores to each of the k bands
        # The total is sum_{k=1 to x} C(x, k) * S(n, k) * y^k
        # where S(n, k) is the Stirling numbers of the second kind
        
        # Precompute Stirling numbers of the second kind S(n, k)
        # S(n, k) is the number of ways to partition n distinct objects into k non-empty subsets
        # We can compute S(n, k) using dynamic programming
        S = [[0] * (x+1) for _ in range(n+1)]
        S[0][0] = 1
        for i in range(1, n+1):
            for j in range(1, x+1):
                S[i][j] = (S[i-1][j-1] + j * S[i-1][j]) % MOD
        
        # Precompute combinations C(x, k)
        # We can compute C(x, k) using the formula C(x, k) = x! / (k! * (x-k)!)
        # Precompute factorials and inverse factorials modulo MOD
        fact = [1] * (x+1)
        inv_fact = [1] * (x+1)
        for i in range(1, x+1):
            fact[i] = fact[i-1] * i % MOD
        inv_fact[x] = pow(fact[x], MOD-2, MOD)
        for i in range(x-1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        def comb(a, b):
            if b < 0 or b > a:
                return 0
            return fact[a] * inv_fact[b] % MOD * inv_fact[a-b] % MOD
        
        # Calculate the total number of ways to assign scores
        total_score_assignments = 0
        for k in range(1, x+1):
            c = comb(x, k)
            s = S[n][k]
            y_pow_k = pow(y, k, MOD)
            total_score_assignments = (total_score_assignments + c * s % MOD * y_pow_k) % MOD
        
        # The total number of ways is the product of stage assignments and score assignments
        total_ways = stage_assignments * total_score_assignments % MOD
        return total_ways