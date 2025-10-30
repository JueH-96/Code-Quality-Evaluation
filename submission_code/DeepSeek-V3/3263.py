class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        # Iterate over possible positions for the first and second cuts
        for i in range(1, n-1):
            for j in range(i+1, n):
                # Calculate the cost of the three subarrays
                cost = nums[0] + nums[i] + nums[j]
                if cost < min_cost:
                    min_cost = cost
        
        return min_cost