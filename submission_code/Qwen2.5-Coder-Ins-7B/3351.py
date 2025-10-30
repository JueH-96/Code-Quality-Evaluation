class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        total_happiness = 0
        for i in range(k):
            if happiness[i] - i > 0:
                total_happiness += happiness[i] - i
        return total_happiness