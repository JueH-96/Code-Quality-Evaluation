from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        def count_set_bits(n: int) -> int:
            """Count the number of set bits in the binary representation of n."""
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count

        total_sum = 0
        for i, num in enumerate(nums):
            # Check if the index has exactly k set bits
            if count_set_bits(i) == k:
                total_sum += num

        return total_sum