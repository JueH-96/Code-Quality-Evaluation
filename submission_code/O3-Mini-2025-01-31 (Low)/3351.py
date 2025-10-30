from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order.
        # Picking the k largest ones first minimizes the decrement effect.
        happiness.sort(reverse=True)
        
        max_sum = 0
        for i in range(k):
            # The effective happiness is decreased by i turns.
            # It should not be negative.
            max_sum += max(happiness[i] - i, 0)
        
        return max_sum