class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_cost = [float('inf')] * n
        for i in range(n):
            cost = [0] * n
            min_op = float('inf')
            for j in range(n):
                cost[j] = min(cost[j-1], nums[(i+j)%n])
                min_op = min(min_op, cost[j] + j * x)
            min_cost[i] = min_op
        return min(min_cost)