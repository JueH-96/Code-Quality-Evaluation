from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        total_happiness = 0
        for i in range(k):
            total_happiness += max(0, happiness[i] - i)
        return total_happiness