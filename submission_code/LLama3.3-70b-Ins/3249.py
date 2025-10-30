from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Calculate the XOR of all elements in the array
        xor = 0
        for num in nums:
            xor ^= num
        
        # If the XOR is already equal to k, return 0
        if xor == k:
            return 0
        
        # Calculate the number of operations required to make the XOR equal to k
        operations = 0
        for num in nums:
            # Calculate the XOR of the current number and the target XOR
            target_xor = xor ^ num
            
            # Calculate the number of bits that need to be flipped to make the XOR equal to k
            bits_to_flip = bin(target_xor ^ k).count('1')
            
            # Update the minimum number of operations
            operations = min(operations + bits_to_flip, operations) if operations != 0 else bits_to_flip
        
        # Return the minimum number of operations
        return operations