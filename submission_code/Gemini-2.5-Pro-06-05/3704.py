from typing import List

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        """
        Calculates the number of partitions of an array where the difference
        between the sum of the left and right subarrays is even.

        Args:
            nums: A list of integers.

        Returns:
            The number of valid partitions.
        """
        
        # Let S_left be the sum of the left subarray and S_right be the sum of the right subarray.
        # Let S_total be the sum of the entire array.
        # We have S_left + S_right = S_total.
        #
        # We need the difference S_left - S_right to be even.
        #
        # Substitute S_right = S_total - S_left into the difference:
        # Difference = S_left - (S_total - S_left) = 2 * S_left - S_total.
        #
        # The term 2 * S_left is always even.
        # So, the parity of the difference depends only on the parity of S_total.
        # - If S_total is even, Difference = (even) - (even) = even.
        # - If S_total is odd,  Difference = (even) - (odd)  = odd.
        #
        # Thus, the difference is even if and only if the total sum of the array is even.

        total_sum = sum(nums)
        n = len(nums)

        if total_sum % 2 == 0:
            # If the total sum is even, every possible partition is valid.
            # The number of partitions is n - 1 (for indices 0 to n-2).
            return n - 1
        else:
            # If the total sum is odd, no partition will result in an even difference.
            return 0