from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        
        # Iterate through the array to find all possible starting positions
        # for a subarray of length 3.
        # If a subarray starts at index i, it includes nums[i], nums[i+1], and nums[i+2].
        # Therefore, the loop for 'i' must ensure that 'i+2' is a valid index.
        # The last valid index for 'i' would be len(nums) - 3.
        # So, the range for 'i' is from 0 up to len(nums) - 3 (inclusive).
        # In Python, range(len(nums) - 2) achieves this (stops before len(nums) - 2).
        for i in range(len(nums) - 2):
            # Extract the three numbers for the current subarray
            first_num = nums[i]
            second_num = nums[i+1]
            third_num = nums[i+2]

            # Check the condition: sum of first and third numbers equals half of the second number.
            # To avoid floating-point arithmetic and potential precision issues,
            # we transform the condition: 2 * (first_num + third_num) == second_num.
            # This also implicitly handles cases where second_num is odd (an even number
            # cannot equal an odd number, so the condition will correctly be false).
            if 2 * (first_num + third_num) == second_num:
                count += 1
                
        return count