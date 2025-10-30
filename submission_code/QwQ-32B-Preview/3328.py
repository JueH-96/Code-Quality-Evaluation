import math

class Solution:
    def minOperations(self, k: int) -> int:
        if k <= 1:
            return 0
        
        sqrt_k = math.isqrt(k)
        candidates = [sqrt_k - 1, sqrt_k, sqrt_k + 1, sqrt_k + 2]
        candidates = [x for x in candidates if x >= 1]
        
        min_ops = float('inf')
        
        for x in candidates:
            y = math.ceil(k / x) - 1
            ops = (x - 1) + y
            if ops < min_ops:
                min_ops = ops
        
        return min_ops