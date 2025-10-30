from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        answer = []
        current_min = float('inf')
        for c in cost:
            current_min = min(current_min, c)
            answer.append(current_min)
        return answer