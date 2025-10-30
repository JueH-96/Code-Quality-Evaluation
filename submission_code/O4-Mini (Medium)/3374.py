from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        # next_len will store the length of the maximal alternating subarray
        # starting at the current index i.
        next_len = 0
        # Traverse from the end to the beginning.
        for i in range(n - 1, -1, -1):
            # If at the last element or the current and next are equal,
            # we can only start a new alternating subarray of length 1 here.
            if i == n - 1 or nums[i] == nums[i + 1]:
                next_len = 1
            else:
                # Otherwise, extend the alternating length by 1.
                next_len += 1
            total += next_len
        return total