class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # Calculate the effective number of passes after k seconds
        effective_passes = k % (n - 1)
        
        # Determine the child who receives the ball after k seconds
        if k < n:
            return k
        elif k % (n - 1) == 0:
            return n - 1
        elif effective_passes <= (n - 1) // 2:
            return effective_passes
        else:
            return n - 1 - (effective_passes - (n - 1) // 2) * 2