class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Calculate the overall AND of the entire array
        overall_and = nums[0]
        for num in nums[1:]:
            overall_and &= num
        
        if overall_and != 0:
            # If overall AND is not zero, return 1 as we can't split further without increasing the sum
            return 1
        else:
            count = 0
            running_and = nums[0]
            for num in nums[1:]:
                running_and &= num
                if running_and == 0:
                    count += 1
                    running_and = num  # Reset running_and for the next subarray
            # After the loop, if running_and is zero, consider it as one subarray
            if running_and == 0:
                count += 1
            return count