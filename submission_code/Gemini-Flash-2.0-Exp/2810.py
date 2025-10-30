class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        ans = float('inf')
        for k in range(n):
            cost = k * x
            current_cost = 0
            for i in range(n):
                current_cost += nums[(i - k) % n]
            cost += current_cost
            ans = min(ans, cost)
        return ans