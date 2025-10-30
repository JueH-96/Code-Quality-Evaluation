class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 3:
            return sum(nums)
        
        min_cost = float('inf')
        
        for i in range(1, n-1):
            for j in range(i+1, n):
                cost = nums[0] + nums[i] + nums[j]
                min_cost = min(min_cost, cost)
        
        return min_cost