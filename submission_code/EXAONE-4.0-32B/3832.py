from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        min_val = float('inf')
        ans = []
        for c in cost:
            if c < min_val:
                min_val = c
            ans.append(min_val)
        return ans