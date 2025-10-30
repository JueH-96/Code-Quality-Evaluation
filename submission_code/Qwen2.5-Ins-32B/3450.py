class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # Calculate the number of full cycles and the remaining steps
        full_cycles, remaining_steps = divmod(k, 2 * (n - 1))
        
        # Determine the position based on the remaining steps
        if remaining_steps < n:
            return remaining_steps
        else:
            return 2 * n - 2 - remaining_steps