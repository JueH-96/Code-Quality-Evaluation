class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        cost = nums[:]
        min_total_cost = sum(cost)
        for t in range(1, n):
            for k in range(n):
                cost[k] = min(cost[k], nums[(k + t)%n])
            total_cost = t * x + sum(cost)
            min_total_cost = min(min_total_cost, total_cost)
        return min_total_cost