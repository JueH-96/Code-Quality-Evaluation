class Solution:
    def minOperations(self, k: int) -> int:
        # If k is 1 or less, the sum is already >= k with the initial array [1],
        # so no operations are needed.
        if k <= 1:
            return 0
        
        # We'll try all possible final "largest element" values x from 1 to k.
        # 1) We need (x-1) increments to get from 1 to x.
        # 2) We then need to duplicate x enough times so that n*x >= k,
        #    i.e. n = ceil(k/x).
        #    That requires (n-1) duplication operations.
        # Hence the total cost is (x - 1) + (n - 1).
        
        import math
        
        best = float('inf')
        for x in range(1, k + 1):
            # Number of copies needed:
            n = (k + x - 1) // x  # equivalently math.ceil(k/x)
            
            # Total operations:
            cost = (x - 1) + (n - 1)
            best = min(best, cost)
        
        return best