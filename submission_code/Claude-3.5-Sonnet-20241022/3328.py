class Solution:
    def minOperations(self, k: int) -> int:
        if k <= 1:
            return 0
            
        min_ops = float('inf')
        
        # Try different numbers of elements in final array
        for num in range(1, k + 1):
            # For each number, calculate value needed per element
            val_needed = (k + num - 1) // num  # Ceiling division
            
            # Operations needed:
            # 1. Increase initial 1 to val_needed: val_needed - 1
            # 2. Duplicate to get num elements: num - 1
            ops = (val_needed - 1) + (num - 1)
            min_ops = min(min_ops, ops)
            
            # Optimization: if current ops exceeds min_ops, future iterations will too
            if ops > min_ops:
                break
                
        return min_ops