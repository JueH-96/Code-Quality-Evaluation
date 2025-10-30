from typing import List
import itertools

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        n = len(happiness)
        happiness = [0] + happiness
        max_sum = 0
        for i in range(1, n + 1):
            for comb in itertools.combinations(range(1, n + 1), i):
                selected = [0] * (n + 1)
                for j in comb:
                    selected[j] = 1
                happiness_sum = sum(happiness[i] * selected[j] for i, j in enumerate(range(1, n + 1)))
                if happiness_sum > max_sum:
                    max_sum = happiness_sum
        return max_sum