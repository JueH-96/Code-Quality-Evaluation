from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness array in descending order
        sorted_happiness = sorted(happiness, reverse=True)
        # Initialize the sum of happiness
        total_sum = 0
        # Iterate through the first k elements and calculate the happiness with decrement
        for i in range(k):
            # Add the maximum of 0 and the current happiness minus the number of selections so far
            total_sum += max(0, sorted_happiness[i] - i)
        return total_sum