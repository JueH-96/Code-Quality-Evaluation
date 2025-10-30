class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # Calculate the effective number of passes
        effective_passes = k % (2 * n - 2)
        
        # Determine the final position of the ball
        if effective_passes < n:
            return effective_passes
        else:
            return 2 * n - 2 - effective_passes