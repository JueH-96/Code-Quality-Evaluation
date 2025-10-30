class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        # Try all possible ways to split array into 3 parts
        for i in range(1, n-1):  # First cut position
            for j in range(i+1, n):  # Second cut position
                # Three subarrays: nums[0:i], nums[i:j], nums[j:n]
                cost = nums[0] + nums[i] + nums[j]
                min_cost = min(min_cost, cost)
                
        return min_cost