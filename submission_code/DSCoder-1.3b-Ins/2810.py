class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        cost = [0]*n
        for i in range(n):
            cost[i] = nums[i]
        cost.sort()
        total_cost = 0
        for i in range(n):
            total_cost += cost[i]
            if i < n-1:
                x -= 1
                if x < 0:
                    return -1
        if x > 0:
            return -1
        return total_cost