class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        total_sum = 0
        n = len(nums)
        
        for i in range(n):
            # Calculate the starting index for the subarray
            start = max(0, i - nums[i])
            
            # Calculate the sum of the subarray from start to i (inclusive)
            subarray_sum = sum(nums[start:i+1])
            
            # Add to the total sum
            total_sum += subarray_sum
        
        return total_sum