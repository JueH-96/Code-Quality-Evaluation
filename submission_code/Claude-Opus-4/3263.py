class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        # Try all possible ways to split into 3 subarrays
        # i is the start of the second subarray
        # j is the start of the third subarray
        for i in range(1, n-1):
            for j in range(i+1, n):
                # Cost is the first element of each subarray
                cost = nums[0] + nums[i] + nums[j]
                min_cost = min(min_cost, cost)
        
        return min_cost