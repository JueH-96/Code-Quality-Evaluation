from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        """
        Iterate over nums together with their indices.
        For each index, count the number of 1-bits in its binary form.
        If that count equals k, add the element at that index to the running sum.
        """
        total = 0
        for idx, val in enumerate(nums):
            # Count set bits in idx.
            # Using int.bit_count() if available, else fall back to bin().count('1')
            try:
                set_bits = idx.bit_count()
            except AttributeError:                 # For Python versions < 3.8
                set_bits = bin(idx).count('1')
            
            if set_bits == k:
                total += val
        return total