class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        min_val = float('inf')
        res = []
        for c in cost:
            if c < min_val:
                min_val = c
            res.append(min_val)
        return res