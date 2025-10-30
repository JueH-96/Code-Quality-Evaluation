from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # First, compute the XOR of all elements.
        totalXor = 0
        for num in nums:
            totalXor ^= num
        
        # Notice that each operation flips exactly one bit in one element.
        # Flipping a bit toggles that bit in the element. Since the overall XOR
        # is computed bitwise, we can change each bit of the XOR independently.
        #
        # If we let m = totalXor, we want m (after operations) to become k.
        # Changing one bit in one element is a one‐operation way to flip the corresponding 
        # bit in the overall XOR. It turns out that the minimum number of operations 
        # required is exactly the number of bit positions for which m and k differ.
        #
        # In other words, we need to count the number of 1’s in the binary representation of (m XOR k).
        # This is the Hamming distance between m and k.
        
        return bin(totalXor ^ k).count("1")