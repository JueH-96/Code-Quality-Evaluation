class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Special case: if we need n-1 equal adjacent pairs, all elements must be same
        if k == n - 1:
            return m
        
        # Calculate C(n-1, k) using modular arithmetic
        def mod_inverse(a, mod):
            return pow(a, mod - 2, mod)
        
        # Calculate C(n-1, k) = (n-1)! / (k! * (n-1-k)!)
        numerator = 1
        denominator = 1
        
        # Calculate (n-1)! / (n-1-k)! = (n-1) * (n-2) * ... * (n-k)
        for i in range(n - k, n):
            numerator = (numerator * i) % MOD
        
        # Calculate k!
        for i in range(1, k + 1):
            denominator = (denominator * i) % MOD
        
        # C(n-1, k) = numerator / denominator
        combinations = (numerator * mod_inverse(denominator, MOD)) % MOD
        
        # Number of ways to assign values to segments
        # m choices for first segment, (m-1) choices for each of the remaining (n-k-1) segments
        value_assignments = (m * pow(m - 1, n - k - 1, MOD)) % MOD
        
        return (combinations * value_assignments) % MOD