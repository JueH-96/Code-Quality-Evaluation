class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_costs = [float('inf')] * n
        ans = float('inf')
        for k in range(n):
            for j in range(n):
                idx = (j - k) % n
                if nums[idx] < min_costs[j]:
                    min_costs[j] = nums[idx]
            total_cost = k * x + sum(min_costs)
            if total_cost < ans:
                ans = total_cost
        return ans