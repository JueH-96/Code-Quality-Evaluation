from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        ans = []
        cur_min = float('inf')
        for c in cost:
            cur_min = min(cur_min, c)
            ans.append(cur_min)
        return ans