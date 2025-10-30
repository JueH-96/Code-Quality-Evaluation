class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        # Try all possible ways to split into 3 subarrays
        for i in range(1, n-1):
            for j in range(i+1, n):
                cost = nums[0] + nums[i] + nums[j]
                min_cost = min(min_cost, cost)
        
        return min_cost