class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        total = pow(26, n, MOD)
        
        # Using inclusion-exclusion principle
        # A: strings with 0 'l's
        # B: strings with < 2 'e's (0 or 1 'e')  
        # C: strings with 0 't's
        
        A = pow(25, n, MOD)
        B = (pow(25, n, MOD) + n * pow(25, n-1, MOD)) % MOD
        C = pow(25, n, MOD)
        
        AB = (pow(24, n, MOD) + n * pow(24, n-1, MOD)) % MOD
        AC = pow(24, n, MOD)  
        BC = (pow(24, n, MOD) + n * pow(24, n-1, MOD)) % MOD
        
        ABC = (pow(23, n, MOD) + n * pow(23, n-1, MOD)) % MOD
        
        # Apply inclusion-exclusion: |U| - |A ∪ B ∪ C|
        result = (total - A - B - C + AB + AC + BC - ABC) % MOD
        
        return result