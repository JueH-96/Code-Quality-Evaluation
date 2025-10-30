class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        result = 0
        current_length = 1  # Current alternating subarray length
        
        # First element always contributes 1 subarray
        result += 1
        
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                # Extend the current alternating subarray
                current_length += 1
            else:
                # Start a new alternating subarray
                current_length = 1
            
            # Add the number of subarrays ending at position i
            result += current_length
        
        return result