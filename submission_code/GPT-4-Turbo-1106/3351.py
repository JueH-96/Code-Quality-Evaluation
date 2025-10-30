class Solution:
    def maximumHappinessSum(self, happiness: list[int], k: int) -> int:
        happiness.sort(reverse=True)
        sum_happiness = 0
        for i in range(k):
            if happiness[i] > 0:
                sum_happiness += happiness[i]
                for j in range(i+1, len(happiness)):
                    if happiness[j] > 0:
                        happiness[j] -= 1
        return sum_happiness