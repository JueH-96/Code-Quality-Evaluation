from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        The XOR of the whole array changes exactly in the bit that is flipped in a single
        element.  Therefore, one operation toggles exactly one bit in the global XOR.
        
        To transform the current array XOR (`x`) into the target `k` we must toggle every
        bit in which `x` and `k` differ.  The number of such bits is the pop-count of
        `x ^ k`, and it is both necessary (each differing bit must be toggled at least
        once) and sufficient (we can toggle each of them independently).
        
        So the minimum number of operations equals the number of 1-bits in `x ^ k`.
        """
        xor_sum = 0
        for num in nums:
            xor_sum ^= num                 # XOR of the entire array
        
        # Bits that differ between current XOR and k
        diff = xor_sum ^ k
        
        # Number of differing bits = minimum operations
        return diff.bit_count()            # popcount in Python ≥ 3.8