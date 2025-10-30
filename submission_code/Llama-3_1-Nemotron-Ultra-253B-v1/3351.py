from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        total = 0
        for i in range(k):
            contribution = happiness[i] - i
            if contribution > 0:
                total += contribution
        return total