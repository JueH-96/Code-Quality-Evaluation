class Solution:
    def minOperations(self, k: int) -> int:
        import math
        
        # If k <= 1, the initial array [1] already has sum >= k, so no operations needed.
        if k <= 1:
            return 0
        
        ans = float('inf')
        # We'll search around the square root of k for "x" and "n"
        # such that x*n >= k, and minimize (x - 1) + (n - 1) = x + n - 2.
        limit = int(math.isqrt(k)) + 2
        
        for i in range(1, limit):
            # Case 1: let x = i, then n = ceil(k/x)
            x = i
            n = (k + x - 1) // x  # integer ceil
            ans = min(ans, x + n - 2)
            
            # Case 2: let n = i, then x = ceil(k/n)
            n = i
            x = (k + n - 1) // n  # integer ceil
            ans = min(ans, x + n - 2)
        
        return ans