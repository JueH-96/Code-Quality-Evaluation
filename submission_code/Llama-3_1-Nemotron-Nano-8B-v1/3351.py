from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness array in descending order
        happiness.sort(reverse=True)
        # Calculate the sum of max(happiness[i] - i, 0) for the first k elements
        total = sum(max(happiness[i] - i, 0) for i in range(k))
        return total