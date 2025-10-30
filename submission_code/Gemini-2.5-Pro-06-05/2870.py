from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        """
        Finds the maximum length of an alternating subarray in the given list of numbers.

        An alternating subarray `s` of length `m > 1` must satisfy:
        1. s[1] = s[0] + 1
        2. The pattern continues as [s[0], s[1], s[0], s[1], ...], which means for any
           k >= 2, s[k] must be equal to s[k-2].

        This solution iterates through the array once, maintaining the length of the
        current alternating subarray ending at the current position. This allows for
        an efficient O(n) time complexity and O(1) space complexity solution.
        """
        n = len(nums)
        max_len = -1
        # current_len tracks the length of the alternating subarray ending at the previous index (i-1).
        current_len = 0
        
        for i in range(1, n):
            # Case 1: Check if we can extend the current alternating subarray.
            # This is possible if we have a valid subarray (current_len > 0) and
            # the current element nums[i] matches the element two positions back in the array,
            # which would be the start of the last pair in the alternating sequence.
            if current_len > 0 and nums[i] == nums[i - 2]:
                current_len += 1
            # Case 2: If we can't extend, check if a new alternating subarray starts at i-1.
            # This happens if nums[i] is one greater than the previous element.
            elif nums[i] == nums[i - 1] + 1:
                current_len = 2
            # Case 3: If neither condition is met, the alternating pattern is broken.
            else:
                current_len = 0
            
            # Update the overall maximum length found so far.
            max_len = max(max_len, current_len)
            
        return max_len