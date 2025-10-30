from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        # XOR of current_xor and k will give the bits that need to be flipped
        target_xor = current_xor ^ k
        
        # Count the number of 1s in the binary representation of target_xor
        return bin(target_xor).count('1')