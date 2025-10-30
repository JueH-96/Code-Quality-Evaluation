from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order.
        # This way, the highest values can be picked in the earliest turns
        # to suffer the minimal reduction.
        happiness.sort(reverse=True)
        
        total = 0
        # For each turn (0-indexed), the effective happiness of the selected child
        # is max(0, original_value - turn_index).
        for i in range(k):
            total += max(0, happiness[i] - i)
        
        return total