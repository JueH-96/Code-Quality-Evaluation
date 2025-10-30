from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        """
        The cheapest way to stand at position i (0-based) is to pay the
        cheapest person among the first i + 1 people once and then, if we
        overshot the desired position, move backwards by free swaps.

        Thus answer[i] is simply the minimum value in cost[0 … i].
        """
        answer = []
        current_min = float('inf')
        for c in cost:
            current_min = min(current_min, c)
            answer.append(current_min)
        return answer