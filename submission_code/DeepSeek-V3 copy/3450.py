class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # Calculate the effective steps after considering the full cycle
        cycle_length = 2 * (n - 1)
        effective_k = k % cycle_length
        
        # Determine the position based on the effective steps
        if effective_k < n:
            return effective_k
        else:
            return 2 * (n - 1) - effective_k