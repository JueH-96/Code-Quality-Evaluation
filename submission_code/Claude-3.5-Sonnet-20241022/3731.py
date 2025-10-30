class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        
        # Iterate through each index
        for i in range(n):
            # Calculate start index: max(0, i - nums[i])
            start = max(0, i - nums[i])
            
            # Calculate sum of subarray from start to i
            curr_sum = sum(nums[start:i+1])
            
            # Add to total sum
            total_sum += curr_sum
            
        return total_sum