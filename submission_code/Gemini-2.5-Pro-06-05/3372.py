from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        """
        Calculates the length of the longest strictly increasing or strictly decreasing subarray.

        Args:
            nums: A list of integers.

        Returns:
            The length of the longest monotonic subarray.
        """
        n = len(nums)
        if n <= 1:
            return n

        # max_len stores the maximum length found so far.
        # inc_len stores the length of the current strictly increasing subarray.
        # dec_len stores the length of the current strictly decreasing subarray.
        # All are initialized to 1, as a single element is a valid subarray of length 1.
        max_len = 1
        inc_len = 1
        dec_len = 1

        for i in range(1, n):
            # Compare the current element with the previous one to update streak lengths.
            if nums[i] > nums[i - 1]:
                # The increasing streak continues.
                inc_len += 1
                # The decreasing streak is broken, reset it.
                dec_len = 1
            elif nums[i] < nums[i - 1]:
                # The decreasing streak continues.
                dec_len += 1
                # The increasing streak is broken, reset it.
                inc_len = 1
            else:
                # If elements are equal, both strictly monotonic streaks are broken.
                inc_len = 1
                dec_len = 1
            
            # Update the overall maximum length found so far.
            max_len = max(max_len, inc_len, dec_len)
            
        return max_len