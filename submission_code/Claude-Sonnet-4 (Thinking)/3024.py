class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        MOD = 10**9 + 7
        
        # Check if t is a rotation of s
        if t not in s + s:
            return 0
        
        # Find all rotation amounts that result in t
        target_rotations = []
        for i in range(n):
            if s[i:] + s[:i] == t:
                target_rotations.append(i)
        
        def power(base, exp, mod):
            result = 1
            base = base % mod
            while exp > 0:
                if exp % 2 == 1:
                    result = (result * base) % mod
                exp = exp >> 1
                base = (base * base) % mod
            return result
        
        def mod_inverse(a, mod):
            return power(a, mod - 2, mod)
        
        n_inv = mod_inverse(n, MOD)
        
        total = 0
        for target_rotation in target_rotations:
            if target_rotation == 0:
                # Formula for reaching rotation 0
                if k % 2 == 0:
                    term1 = (n - 1) % MOD
                else:
                    term1 = (MOD - (n - 1) % MOD) % MOD
                term2 = power(n - 1, k, MOD)
                numerator = (term1 + term2) % MOD
                result = (numerator * n_inv) % MOD
            else:
                # Formula for reaching rotation i != 0
                if (k + 1) % 2 == 0:
                    term1 = 1
                else:
                    term1 = MOD - 1
                term2 = power(n - 1, k, MOD)
                numerator = (term1 + term2) % MOD
                result = (numerator * n_inv) % MOD
            
            total = (total + result) % MOD
        
        return total