from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)

        # Step 1: Compute the last occurrence index for each unique number
        # This dictionary will store the last index where each number appears.
        last_occurrence = {}
        for i in range(n):
            last_occurrence[nums[i]] = i

        # Step 2 & 3: Initialize variables
        # num_allowed_cuts counts the number of indices i (0 <= i < n-1) after which
        # a cut can be made.
        # max_last tracks the maximum last occurrence index among the numbers encountered
        # in nums[0...i] so far.
        num_allowed_cuts = 0
        max_last = -1 # Initialize with -1. The first element's last_occurrence will be >= 0.

        # Step 4-5: Iterate through potential cut locations (after index i, for i from 0 to n-2)
        # A cut after index i is allowed if the segment nums[0...i] contains all occurrences
        # of the numbers present in nums[0...i].
        # This condition is met if and only if the maximum last occurrence index among
        # numbers in nums[0...i] is less than or equal to i.
        # If this condition is met, it means the prefix nums[0...i] can form a valid
        # self-contained subarray. Any number appearing after index i must have its
        # first occurrence after index i (otherwise, its last occurrence would be > i,
        # contradicting i >= max_last). This ensures that the first subarray and the
        # rest of the array do not share elements.

        for i in range(n - 1):
            # Update max_last based on the current element nums[i]. The maximum last
            # occurrence index seen so far for numbers in nums[0...i] is the maximum
            # of the previous max_last and the last occurrence of nums[i].
            max_last = max(max_last, last_occurrence[nums[i]])

            # If the current index i is greater than or equal to the maximum last occurrence
            # found among numbers in nums[0...i], it means the segment nums[0...i]
            # contains all occurrences of all numbers present within it.
            # This location (after index i) is therefore a valid place to make a cut
            # to end a subarray.
            if i >= max_last:
                 num_allowed_cuts += 1

        # Step 6: The K allowed cut locations (0-indexed) effectively divide the array
        # into K+1 fundamental blocks (segments). A good partition can be formed by
        # choosing any subset of these K allowed cut locations as the actual cut boundaries.
        # Each allowed cut location can either be a cut boundary in the final partition or not.
        # The number of ways to choose a subset of K items is 2^K.
        # We compute 2^num_allowed_cuts modulo 10^9 + 7.

        # Python's built-in pow() handles modular exponentiation efficiently.
        result = pow(2, num_allowed_cuts, mod)

        return result