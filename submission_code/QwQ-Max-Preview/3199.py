class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def comb2(k):
            if k < 2:
                return 0
            return k * (k - 1) // 2
        
        t = comb2(n + 2)
        
        # Compute the term for one child exceeding (A)
        a = 0
        rem_a = n - (limit + 1)
        if rem_a >= 0:
            a = comb2(rem_a + 2)
        a *= 3  # Since there are three children
        
        # Compute the term for two children exceeding (AB)
        ab = 0
        rem_ab = n - 2 * (limit + 1)
        if rem_ab >= 0:
            ab = comb2(rem_ab + 2)
        ab *= 3  # Three pairs (x,y), (x,z), (y,z)
        
        # Compute the term for all three children exceeding (ABC)
        abc = 0
        rem_abc = n - 3 * (limit + 1)
        if rem_abc >= 0:
            abc = comb2(rem_abc + 2)
        
        total = t - a + ab - abc
        return max(total, 0)