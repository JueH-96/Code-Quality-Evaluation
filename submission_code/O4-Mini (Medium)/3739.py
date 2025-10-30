class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        total = m * n
        # If fewer than 2 pieces, distance sum is zero (though constraints guarantee k>=2).
        if k < 2:
            return 0
        
        # Precompute factorials and inverse factorials up to total.
        # We'll need C(total-2, k-2).
        maxN = total
        fact = [1] * (maxN + 1)
        for i in range(1, maxN + 1):
            fact[i] = fact[i-1] * i % MOD
        
        invfact = [1] * (maxN + 1)
        invfact[maxN] = pow(fact[maxN], MOD-2, MOD)
        for i in range(maxN, 0, -1):
            invfact[i-1] = invfact[i] * i % MOD
        
        def comb(n: int, r: int) -> int:
            if r < 0 or r > n:
                return 0
            return fact[n] * invfact[r] % MOD * invfact[n-r] % MOD
        
        # Number of ways to choose the other k-2 pieces once two specific cells are chosen:
        choose_rest = comb(total-2, k-2)
        
        # Precompute modular inverse of 6 for combination formula.
        inv6 = pow(6, MOD-2, MOD)
        
        # Sum of vertical distances over all unordered pairs of cells:
        # sum_{d=1..m-1} d * (m-d) = C(m+1, 3)
        Cm13 = (m * (m-1) % MOD) * (m+1) % MOD * inv6 % MOD
        vert_sum = (n * n) % MOD * Cm13 % MOD
        
        # Sum of horizontal distances over all unordered pairs of cells:
        # sum_{d=1..n-1} d * (n-d) = C(n+1, 3)
        Cn13 = (n * (n-1) % MOD) * (n+1) % MOD * inv6 % MOD
        horiz_sum = (m * m) % MOD * Cn13 % MOD
        
        # Total sum of distances over all unordered pairs of cells:
        total_pair_dist = (vert_sum + horiz_sum) % MOD
        
        # Multiply by the number of ways to pick the remaining k-2 pieces:
        ans = total_pair_dist * choose_rest % MOD
        return ans