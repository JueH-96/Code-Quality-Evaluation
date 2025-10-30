from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order
        happiness.sort(reverse=True)
        
        # Initialize the sum of happiness values
        total_happiness = 0
        
        # Select k children
        for i in range(k):
            # Add the happiness value of the current child to the total happiness
            total_happiness += max(0, happiness[i] - i)
        
        # Return the maximum sum of happiness values
        return total_happiness