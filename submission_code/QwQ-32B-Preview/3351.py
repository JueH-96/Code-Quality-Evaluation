class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness array in descending order
        happiness_sorted = sorted(happiness, reverse=True)
        
        # Calculate the sum of happiness values, adjusting for the turn number
        total = 0
        for i in range(k):
            adjusted_happiness = happiness_sorted[i] - i
            total += max(adjusted_happiness, 0)
        
        return total