class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        count = 0
        current_and = -1  # Initialize with all bits set to 1
        
        for num in nums:
            if current_and == -1:
                current_and = num
            else:
                current_and &= num
            
            # If current AND becomes 0, we can end this subarray
            if current_and == 0:
                count += 1
                current_and = -1  # Reset for next subarray
        
        # If we couldn't form any subarray with AND = 0,
        # we must keep the entire array as one subarray
        return max(1, count)