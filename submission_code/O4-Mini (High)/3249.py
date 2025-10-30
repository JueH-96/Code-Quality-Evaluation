from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Compute current XOR of all elements
        xor_all = 0
        for num in nums:
            xor_all ^= num
        
        # We need to turn xor_all into k.
        # Each bit in which xor_all and k differ must be flipped once.
        # Flipping a bit in any single element toggles that bit in the overall XOR.
        # Hence the minimum number of operations is just the Hamming weight
        # of xor_all ^ k.
        diff = xor_all ^ k
        
        # Python 3.8+ has int.bit_count(); otherwise you can use bin(diff).count('1')
        return diff.bit_count()