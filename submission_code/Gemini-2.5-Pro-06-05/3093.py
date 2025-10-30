from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of elements in nums whose indices have k set bits.

        Args:
            nums: A 0-indexed list of integers.
            k: An integer representing the target number of set bits.

        Returns:
            The sum of elements nums[i] where the index i has k set bits.
        """
        total_sum = 0
        for i, num in enumerate(nums):
            # The int.bit_count() method efficiently counts the number of set bits (1s)
            # in the binary representation of an integer. It is available in Python 3.10+.
            if i.bit_count() == k:
                total_sum += num
        return total_sum