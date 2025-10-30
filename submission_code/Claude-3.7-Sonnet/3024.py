class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        MOD = 10**9 + 7
        
        # Find all rotations of s that equal t
        rotations = []
        doubled = s + s
        for i in range(n):
            if doubled[i:i+n] == t:
                rotations.append(i)
        
        if not rotations:
            return 0
        
        # Calculate using mathematical formula based on matrix properties
        # For a rotation r:
        # If r = 0: ((n-1)^k + (-1)^k*(n-1))/n
        # If r ≠ 0: ((n-1)^k - (-1)^k)/n
        
        result = 0
        for r in rotations:
            if r == 0:
                # Calculate for r = 0 case
                numerator = pow(n-1, k, MOD) + ((-1)**k) * (n-1)
                numerator %= MOD
            else:
                # Calculate for r ≠ 0 case
                numerator = pow(n-1, k, MOD) - ((-1)**k)
                numerator %= MOD
            
            # Calculate n^(-1) mod MOD using Fermat's Little Theorem
            n_inverse = pow(n, MOD-2, MOD)
            ways = (numerator * n_inverse) % MOD
            
            result = (result + ways) % MOD
            
        return result