from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        min_cost = float('inf')
        result = []
        for c in cost:
            min_cost = min(min_cost, c)
            result.append(min_cost)
        return result