class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        max_happiness_sum = 0
        for i in range(k):
            current_happiness = max(0, happiness[i] - i)
            max_happiness_sum += current_happiness
        return max_happiness_sum