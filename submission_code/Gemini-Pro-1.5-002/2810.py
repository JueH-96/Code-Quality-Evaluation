class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        costs = [float('inf')] * n
        for start in range(n):
            current_cost = 0
            for i in range(n):
                costs[i] = min(costs[i], nums[(start + i) % n] + x * i)
        return sum(costs)