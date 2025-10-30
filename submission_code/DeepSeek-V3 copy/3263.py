class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        
        # Iterate over all possible positions for the first and second cuts
        for i in range(1, n-1):
            for j in range(i+1, n):
                # Calculate the sum of the first elements of the three subarrays
                current_sum = nums[0] + nums[i] + nums[j]
                if current_sum < min_sum:
                    min_sum = current_sum
        
        return min_sum