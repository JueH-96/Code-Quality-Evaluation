class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Initialize the count of subarrays and the current AND value
        count = 0
        current_and = nums[0]
        
        for num in nums:
            current_and &= num
            if current_and == 0:
                count += 1
                current_and = num
        
        # If the final AND is not zero, we can only have one subarray
        if current_and != 0:
            return 1
        
        return count