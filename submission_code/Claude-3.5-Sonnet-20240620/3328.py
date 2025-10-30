class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        
        min_ops = float('inf')
        for i in range(1, int(k**0.5) + 1):
            if k % i == 0:
                j = k // i
                ops = i - 1 + j - 1
                min_ops = min(min_ops, ops)
        
        return min_ops