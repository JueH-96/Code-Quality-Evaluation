from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # diff will help us track the net effect of all operations.
        diff = [0] * (n + 1)
        current = 0  # Current accumulated decrements up to index i.
        
        for i in range(n):
            current += diff[i]
            # The remaining decrement needed at index i.
            needed = nums[i] - current
            # If we have over-decremented, it's impossible to fix.
            if needed < 0:
                return False
            # If no decrement is needed, continue.
            if needed == 0:
                continue
            # If applying an operation starting at i would go out of bounds, return False.
            if i + k > n:
                return False
            # Apply 'needed' operations starting from index i.
            current += needed
            # Mark the ending of the effect of these operations at i+k.
            diff[i+k] -= needed
        return True