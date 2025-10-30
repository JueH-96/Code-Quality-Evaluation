class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Initialize result counter and current AND value
        result = 0
        curr_and = -1
        
        # Iterate through array
        for num in nums:
            # For first number in potential subarray
            if curr_and == -1:
                curr_and = num
            else:
                # Take AND with current number
                curr_and &= num
            
            # If current AND is 0, we can split here
            if curr_and == 0:
                result += 1
                curr_and = -1
        
        # If we have a remaining subarray that's not 0
        # or array is empty, return 1
        return max(1, result)