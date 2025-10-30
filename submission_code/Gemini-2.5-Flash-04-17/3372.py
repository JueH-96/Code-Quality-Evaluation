from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        # According to constraints, n >= 1, so the result is at least 1.
        # If n=0 were possible, return 0.
        if n == 0:
            return 0

        # Initialize max length found so far. Minimum possible length is 1 (a single element).
        max_len = 1
        # Track the length of the strictly increasing subarray ending at the current position.
        current_inc_len = 1
        # Track the length of the strictly decreasing subarray ending at the current position.
        current_dec_len = 1

        # Iterate from the second element
        for i in range(1, n):
            if nums[i] > nums[i-1]:
                # Current element is greater than the previous one.
                # The strictly increasing subarray continues. Increment its length.
                current_inc_len += 1
                # The strictly decreasing subarray is broken. Restart its length count from 1.
                current_dec_len = 1
            elif nums[i] < nums[i-1]:
                # Current element is less than the previous one.
                # The strictly decreasing subarray continues. Increment its length.
                current_dec_len += 1
                # The strictly increasing subarray is broken. Restart its length count from 1.
                current_inc_len = 1
            else: # nums[i] == nums[i-1]
                # Current element is equal to the previous one.
                # Both strictly increasing and strictly decreasing sequences are broken.
                # Restart both length counts from 1.
                current_inc_len = 1
                current_dec_len = 1

            # Update the overall maximum length found so far.
            # At each step i, we consider the maximum length ending at i,
            # which is the maximum of the current increasing and decreasing lengths.
            max_len = max(max_len, current_inc_len, current_dec_len)

        return max_len