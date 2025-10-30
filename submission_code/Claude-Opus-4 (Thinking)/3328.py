class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        
        min_ops = k - 1  # Case where we just increase without duplication
        
        # Try different values of x (the value we increase to)
        for x in range(1, k):
            # Calculate minimum number of copies needed
            n = (k + x - 1) // x  # This computes ceil(k/x)
            # Total operations: (x-1) increases + (n-1) duplications
            ops = x - 1 + n - 1
            min_ops = min(min_ops, ops)
        
        return min_ops