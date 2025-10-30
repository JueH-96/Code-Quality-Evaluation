from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # Difference array to simulate the effect of operations on a sliding window.
        diff = [0] * (n + 1)
        running = 0  # Running total of operations affecting the current index.
        
        for i in range(n):
            # Update the running effect with the difference array.
            running += diff[i]
            # The effective value at index i after applying all operations so far.
            effective = nums[i] - running
            # If the effective value is already negative, it is impossible to fix since we can only subtract.
            if effective < 0:
                return False
            
            # The number of additional operations needed at position i.
            need = effective
            # If we can still apply a full window starting at index i then do so.
            if i + k <= n:
                running += need         # These operations now affect the current and following k-1 positions.
                diff[i + k] -= need     # Mark the end of the window's effect.
            # If we can't start a window because it's out of bounds, then the current value must be 0.
            elif need != 0:
                return False
        
        return True