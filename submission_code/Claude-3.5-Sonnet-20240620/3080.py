class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        min_score = nums[0]
        for num in nums:
            min_score &= num
        
        if min_score > 0:
            return 1
        
        count = 0
        current_score = -1  # Initialize with all bits set to 1
        
        for num in nums:
            current_score &= num
            if current_score == 0:
                count += 1
                current_score = -1  # Reset for next subarray
        
        return max(1, count)