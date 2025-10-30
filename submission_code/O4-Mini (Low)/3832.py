from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        """
        For each target position i, the cheapest way is to pay the smallest cost among
        all people at positions 0..i. Once you've paid that person, you can swap backwards
        (towards higher indices) for free until you reach i.
        """
        n = len(cost)
        answer = [0] * n
        current_min = float('inf')
        for i in range(n):
            if cost[i] < current_min:
                current_min = cost[i]
            answer[i] = current_min
        return answer