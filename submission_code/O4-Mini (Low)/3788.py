from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        """
        We want the maximum-sum subarray with all unique elements.
        We can use a sliding window (two pointers) with a set to maintain uniqueness,
        and keep track of the running sum of the window. As we extend the right pointer,
        if we hit a duplicate, we shrink from the left until the duplicate is removed.
        We update the global maximum sum at each step.
        """
        seen = set()
        left = 0
        curr_sum = 0
        max_sum = float('-inf')
        
        for right, val in enumerate(nums):
            # If val is already in the window, shrink from the left
            while val in seen:
                rem = nums[left]
                seen.remove(rem)
                curr_sum -= rem
                left += 1
            
            # Add the new value
            seen.add(val)
            curr_sum += val
            
            # Update max
            if curr_sum > max_sum:
                max_sum = curr_sum
        
        return max_sum