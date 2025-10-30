class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Find the number of rotations needed to transform s to t
        rotations = 0
        for i in range(n):
            if s[i:] + s[:i] == t:
                rotations = i
                break
        
        # If s cannot be transformed to t
        if s[rotations:] + s[:rotations] != t:
            return 0
        
        # Calculate (n-1)^k using binary exponentiation
        def pow_mod(base, exp):
            result = 1
            while exp > 0:
                if exp & 1:
                    result = (result * base) % MOD
                base = (base * base) % MOD
                exp >>= 1
            return result
        
        n_minus_1_pow_k = pow_mod(n-1, k)
        
        # Calculate the final result
        if rotations == 0:
            return (n_minus_1_pow_k + (n-1) * pow_mod(n, k-1)) % MOD if k > 0 else 1
        else:
            return (pow_mod(n, k-1) * (n - 1)) % MOD if k > 0 else 0