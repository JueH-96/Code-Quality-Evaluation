from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # To maximize the sum, we should always pick the currently largest happiness.
        # Each time we pick one child, all remaining children lose 1 happiness (down to a minimum of 0).
        # Equivalently, if we sort happiness in descending order and pick the top k in that order,
        # the i-th pick (0-based) contributes max(h_sorted[i] - i, 0).
        
        # Sort in descending order
        happiness.sort(reverse=True)
        
        total = 0
        # Sum up the contributions for the first k picks
        for i in range(k):
            # Current happiness after i decrements
            curr = happiness[i] - i
            if curr <= 0:
                # Further picks will only be <= 0
                break
            total += curr
        
        return total