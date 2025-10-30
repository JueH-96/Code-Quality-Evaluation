class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort in descending order to always pick the highest available happiness
        happiness.sort(reverse=True)
        
        total_happiness = 0
        
        for i in range(k):
            # After i selections, each child has lost i happiness points
            # Current happiness = original happiness - i, but at least 0
            current_happiness = max(0, happiness[i] - i)
            total_happiness += current_happiness
        
        return total_happiness