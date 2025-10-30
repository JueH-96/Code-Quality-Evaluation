class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Initialize the current AND value to a large number (all bits set)
        current_and = (1 << 20) - 1  # Since nums[i] <= 10^6, 20 bits are enough
        count = 0
        
        for num in nums:
            # Perform AND operation with the current number
            current_and &= num
            
            # If the current AND becomes zero, we can form a subarray
            if current_and == 0:
                count += 1
                # Reset current_and to a large number for the next potential subarray
                current_and = (1 << 20) - 1
        
        # If we never reached a zero AND, we can only have one subarray
        return max(count, 1)