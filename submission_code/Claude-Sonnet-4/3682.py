class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # Calculate C(n-1, k) using the formula: n! / (k! * (n-k)!)
        def combination(n, k):
            if k > n or k < 0:
                return 0
            if k == 0 or k == n:
                return 1
            
            # Calculate C(n, k) = n! / (k! * (n-k)!)
            # We'll calculate this iteratively to avoid overflow
            numerator = 1
            denominator = 1
            
            # Use the property C(n,k) = C(n, n-k) to minimize calculations
            k = min(k, n - k)
            
            for i in range(k):
                numerator = (numerator * (n - i)) % MOD
                denominator = (denominator * (i + 1)) % MOD
            
            # Calculate modular inverse of denominator
            return (numerator * pow(denominator, MOD - 2, MOD)) % MOD
        
        # Calculate (m-1)^(n-1-k)
        def power(base, exp):
            return pow(base, exp, MOD)
        
        # Apply the formula: m * C(n-1, k) * (m-1)^(n-1-k)
        result = m % MOD
        result = (result * combination(n - 1, k)) % MOD
        result = (result * power(m - 1, n - 1 - k)) % MOD
        
        return result