class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        
        import math
        min_ops = float('inf')
        
        # Try all possible values to increase the element to
        for a in range(1, k + 1):
            # Operations to increase 1 to a: (a-1)
            # Operations to duplicate: (ceil(k/a) - 1)
            ops = (a - 1) + (math.ceil(k / a) - 1)
            min_ops = min(min_ops, ops)
        
        return min_ops