class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_length = 1
        current_length = 1
        
        # Check for longest increasing subarray
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                current_length += 1
            else:
                max_length = max(max_length, current_length)
                current_length = 1
        max_length = max(max_length, current_length)
        
        # Reset current_length for decreasing subarray check
        current_length = 1
        
        # Check for longest decreasing subarray
        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                current_length += 1
            else:
                max_length = max(max_length, current_length)
                current_length = 1
        max_length = max(max_length, current_length)
        
        return max_length