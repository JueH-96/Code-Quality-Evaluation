class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort in descending order to select highest happiness values first
        happiness.sort(reverse=True)
        
        total_sum = 0
        for i in range(k):
            # The effective happiness is the original happiness minus the number of previous selections
            # Use max to ensure we don't add negative values
            total_sum += max(0, happiness[i] - i)
            
        return total_sum