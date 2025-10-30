from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Calculate the XOR of all elements in the array
        current_xor = 0
        for num in nums:
            current_xor ^= num

        # Calculate the number of different bits between current_xor and k
        different_bits = bin(current_xor ^ k).count('1')

        return different_bits