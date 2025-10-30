from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Compute current XOR of all elements
        current_xor = 0
        for x in nums:
            current_xor ^= x
        # We need to flip bits so that current_xor becomes k.
        # Each bit flip in any element toggles exactly one bit in the total XOR.
        # To change current_xor to k, we must toggle each bit where they differ.
        # The minimal number of toggles is simply the Hamming distance between current_xor and k.
        diff = current_xor ^ k
        # Count set bits in diff
        return diff.bit_count()