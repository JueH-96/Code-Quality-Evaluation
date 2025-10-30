from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Compute the XOR of all elements
        xor_sum = 0
        for num in nums:
            xor_sum ^= num
        
        # The difference we need to correct
        diff = xor_sum ^ k
        
        # Each bit in diff that is 1 must be flipped once; count those bits
        return diff.bit_count()