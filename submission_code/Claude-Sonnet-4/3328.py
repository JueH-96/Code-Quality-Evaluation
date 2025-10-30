class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        
        min_ops = float('inf')
        
        # Check all reasonable values of x
        for x in range(1, k + 1):
            # Operations to increase 1 to x
            increase_ops = x - 1
            
            # Number of copies needed: ceil(k/x)
            copies_needed = (k + x - 1) // x
            
            # Operations to duplicate (we start with 1 copy)
            duplicate_ops = copies_needed - 1
            
            total_ops = increase_ops + duplicate_ops
            min_ops = min(min_ops, total_ops)
            
            # Early termination: if x >= k, we only need 1 copy
            if x >= k:
                break
        
        return min_ops