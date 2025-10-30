class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness array in descending order
        happiness.sort(reverse=True)
        
        # Initialize the sum of selected happiness values
        max_happiness = 0
        
        # Select the top k elements from the sorted list
        for i in range(k):
            max_happiness += happiness[i]
        
        return max_happiness