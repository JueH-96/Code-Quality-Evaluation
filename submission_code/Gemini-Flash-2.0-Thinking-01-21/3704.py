from typing import List

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        """
        Counts the number of partitions in an array where the difference between
        the sum of the left and right subarrays is even.

        A partition is defined by an index i (0 <= i < n - 1), splitting the array
        into a left part nums[0...i] and a right part nums[i+1...n-1].

        Args:
            nums: The input list of integers.

        Returns:
            The number of valid partitions.
        """
        n = len(nums)

        # Calculate the total sum of the array.
        total_sum = sum(nums)

        # A partition at index i has a left subarray sum S_left = sum(nums[0...i])
        # and a right subarray sum S_right = sum(nums[i+1...n-1]).
        # The condition is that the difference (S_left - S_right) is even.
        # This is equivalent to S_left and S_right having the same parity
        # (both even or both odd).

        # We know that S_left + S_right = total_sum.
        # If S_left and S_right have the same parity:
        # - Even + Even = Even
        # - Odd + Odd = Even
        # In both cases, their sum (total_sum) is even.

        # If S_left and S_right have different parities:
        # - Even + Odd = Odd
        # - Odd + Even = Odd
        # In both cases, their sum (total_sum) is odd.

        # Therefore, S_left and S_right have the same parity if and only if the
        # total_sum of the array is even.

        # Case 1: If the total_sum is odd, it is impossible for S_left and S_right
        # to have the same parity for any partition. Thus, the difference
        # (S_left - S_right) will always be odd. No partitions satisfy the condition.
        if total_sum % 2 != 0:
            return 0
        # Case 2: If the total_sum is even, S_left and S_right will always have
        # the same parity for any partition. Thus, the difference
        # (S_left - S_right) will always be even. All possible partitions satisfy
        # the condition.
        # The possible partition indices i range from 0 to n-2 (inclusive),
        # as the problem specifies 0 <= i < n - 1.
        # The number of such indices (and thus partitions) is (n-2) - 0 + 1 = n - 1.
        else:
            return n - 1