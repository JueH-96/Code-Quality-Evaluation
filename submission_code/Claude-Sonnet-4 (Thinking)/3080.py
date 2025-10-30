class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Calculate the AND of the entire array
        total_and = nums[0]
        for i in range(1, len(nums)):
            total_and &= nums[i]
        
        # If total_and > 0, we can only have 1 subarray
        if total_and > 0:
            return 1
        
        # If total_and == 0, use greedy approach
        count = 0
        current_and = None
        
        for num in nums:
            if current_and is None:
                current_and = num
            else:
                current_and &= num
            
            if current_and == 0:
                count += 1
                current_and = None
        
        # If the last subarray doesn't have AND = 0, return 1
        if current_and is not None:
            return 1
        
        return count