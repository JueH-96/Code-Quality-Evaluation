class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Calculate the AND of the entire array
        total_and = nums[0]
        for i in range(1, len(nums)):
            total_and &= nums[i]
        
        # If the total AND is not 0, we can't split to get a lower score
        if total_and != 0:
            return 1
        
        # Otherwise, greedily split whenever we get a subarray with AND = 0
        count = 0
        current_and = -1  # Initialize with all bits set
        
        for num in nums:
            current_and &= num
            
            if current_and == 0:
                count += 1
                current_and = -1  # Reset for the next subarray
        
        return count