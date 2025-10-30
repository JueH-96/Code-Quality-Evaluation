from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order
        happiness.sort(reverse=True)
        
        # Initialize the sum of happiness values
        max_happiness = 0
        
        # Iterate through the first k elements
        for i in range(k):
            # Decrease the happiness value by the number of turns passed
            if happiness[i] - i > 0:
                max_happiness += happiness[i] - i
        
        return max_happiness