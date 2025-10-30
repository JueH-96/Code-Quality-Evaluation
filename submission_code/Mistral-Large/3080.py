from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Initialize variables to keep track of the minimum AND value and the number of subarrays
        min_and = float('inf')
        current_and = nums[0]
        count = 0

        # Iterate through the array to find the minimum AND value of any subarray
        for num in nums:
            current_and &= num
            min_and = min(min_and, current_and)
            if current_and == 0:
                current_and = float('inf')  # Reset current_and to find the next subarray
                count += 1

        # If min_and is 0, it means we can split the array into subarrays with AND value 0
        if min_and == 0:
            return count
        else:
            return 1