class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness_sorted = sorted(happiness, reverse=True)
        total = 0
        for i in range(k):
            val = happiness_sorted[i] - i
            if val > 0:
                total += val
            else:
                break
        return total