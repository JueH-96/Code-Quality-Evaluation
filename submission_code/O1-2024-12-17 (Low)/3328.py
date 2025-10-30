class Solution:
    def minOperations(self, k: int) -> int:
        # If k <= 1, no operations needed as sum([1]) = 1 >= k
        if k <= 1:
            return 0
        
        # We'll try all possible "final single-element values" x (from 1..k).
        # 1) We need (x - 1) increments to get from 1 to x.
        # 2) Then we duplicate x enough times so that (count of x's) * x >= k.
        #    If we end with e+1 copies of x, we need e duplications (operations),
        #    and (e+1)*x >= k => e+1 >= k/x => e = ceil(k/x) - 1.
        # Total operations = (x - 1) + (ceil(k/x) - 1).
        # We choose the x that yields the minimal total operations.
        
        import math
        
        min_ops = float('inf')
        for x in range(1, k+1):
            # Increments needed = (x - 1)
            increments = x - 1
            # Duplications needed = ceil(k/x) - 1
            # Using integer division trick for ceil: ceil(a/b) = (a + b - 1)//b
            duplications = (k + x - 1) // x - 1
            total = increments + duplications
            if total < min_ops:
                min_ops = total
        
        return min_ops