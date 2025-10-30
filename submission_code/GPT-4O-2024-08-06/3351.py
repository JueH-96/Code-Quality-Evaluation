class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness values in descending order
        happiness.sort(reverse=True)
        
        # Initialize the sum of selected happiness values
        max_happiness_sum = 0
        
        # Select the top k happiness values
        for i in range(k):
            max_happiness_sum += happiness[i]
        
        return max_happiness_sum