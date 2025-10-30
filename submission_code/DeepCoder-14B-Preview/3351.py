class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness_sorted = sorted(happiness, reverse=True)
        total = 0
        for i in range(k):
            total += max(happiness_sorted[i] - i, 0)
        return total