import collections
from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of elements in nums whose corresponding indices
        have exactly k set bits in their binary representation.

        Args:
            nums: A 0-indexed integer array.
            k: An integer representing the required number of set bits.

        Returns:
            An integer denoting the sum of elements meeting the criteria.
        """
        total_sum = 0
        n = len(nums)

        # Iterate through the array using indices
        for i in range(n):
            # Convert the current index 'i' to its binary string representation.
            # For example, bin(0) -> '0b0', bin(1) -> '0b1', bin(2) -> '0b10'.
            # Then, count the number of '1's in this binary string.
            # This efficiently gives the number of set bits.
            set_bits_count = bin(i).count('1')
            
            # If the count of set bits for the current index 'i' is equal to 'k',
            # add the corresponding element nums[i] to the total sum.
            if set_bits_count == k:
                total_sum += nums[i]
                
        return total_sum