class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        total = 0
        for i in range(k):
            reduction = (k - 1) - i
            total += max(0, happiness[i] - reduction)
        return total