from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        # For a circular binary array original, the XOR of all adjacent pairs
        # must be zero: (o0⊕o1)⊕(o1⊕o2)⊕...⊕(o[n-1]⊕o0) = 0 because each o_i cancels out.
        # Hence, we just check if the XOR of all derived elements is 0.
        total_xor = 0
        for bit in derived:
            total_xor ^= bit
        return total_xor == 0