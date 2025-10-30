class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        ans = [cost[0]]
        for num in cost[1:]:
            ans.append(min(ans[-1], num))
        return ans