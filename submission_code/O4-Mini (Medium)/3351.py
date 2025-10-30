from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order
        happiness.sort(reverse=True)
        
        total = 0
        # Pick the top k happiness values in order, accounting for the decrement
        # each turn i (0-based) reduces the value by i, floored at 0.
        for i in range(k):
            # Compute the effective happiness when picked at turn i
            val = happiness[i] - i
            if val <= 0:
                break
            total += val
        
        return total