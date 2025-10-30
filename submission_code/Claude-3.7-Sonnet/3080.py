class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        count = 0
        current_and = -1  # All bits set to 1 (for initial AND operation)
        
        for num in nums:
            current_and &= num
            
            if current_and == 0:
                count += 1
                current_and = -1  # Reset for next subarray
        
        # If we couldn't create any subarray with score 0,
        # then we need to keep the entire array as one subarray
        return max(1, count)