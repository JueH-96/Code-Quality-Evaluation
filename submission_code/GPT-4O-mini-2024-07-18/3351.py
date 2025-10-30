class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order
        happiness.sort(reverse=True)
        
        # Initialize the maximum happiness sum
        max_happiness_sum = 0
        
        # Iterate through the first k children in the sorted happiness list
        for i in range(k):
            # Add the happiness value of the selected child to the sum
            max_happiness_sum += happiness[i]
            # Decrease the happiness of the remaining children
            for j in range(i + 1, n):
                if happiness[j] > 0:
                    happiness[j] -= 1
        
        return max_happiness_sum