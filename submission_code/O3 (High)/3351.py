from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the children by their initial happiness in descending order.
        happiness.sort(reverse=True)

        total = 0
        for turn in range(k):                       # turn goes from 0 to k-1
            current = happiness[turn] - turn        # effective happiness this turn
            if current <= 0:                        # no further child can contribute positively
                break
            total += current

        return total