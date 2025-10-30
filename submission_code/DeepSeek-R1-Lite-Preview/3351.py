class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness array in descending order
        happiness_sorted = sorted(happiness, reverse=True)
        
        total_sum = 0
        for i in range(k):
            # Adjust the happiness value by subtracting the number of turns before selection
            adjusted_happiness = happiness_sorted[i] - i
            if adjusted_happiness < 0:
                adjusted_happiness = 0
            total_sum += adjusted_happiness
        return total_sum