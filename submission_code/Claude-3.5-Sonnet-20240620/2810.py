class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        costs = [nums.copy() for _ in range(n)]
        
        for i in range(1, n):
            for j in range(n):
                costs[i][j] = min(costs[i-1][j], nums[(j-i)%n])
        
        min_cost = float('inf')
        for i in range(n):
            current_cost = i * x + sum(costs[i])
            min_cost = min(min_cost, current_cost)
        
        return min_cost