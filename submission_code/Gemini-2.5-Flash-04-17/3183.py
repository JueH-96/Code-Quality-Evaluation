from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        k_or_result = 0
        
        # We need to check each bit position. Since the numbers are less than 2^31,
        # we only need to consider bits from 0 up to 30.
        # We can iterate from bit_pos 0 to 31 (total 32 bits) to be safe,
        # as standard integers are often 32-bit.
        for bit_pos in range(32):
            count = 0
            
            # Check how many numbers in nums have the bit at bit_pos set
            for num in nums:
                # To check if the bit at bit_pos is set in num:
                # 1. Right shift num by bit_pos: num >> bit_pos
                #    This moves the bit we are interested in to the least significant position.
                # 2. Perform a bitwise AND with 1: (num >> bit_pos) & 1
                #    If the original bit was 1, the result is 1. If the original bit was 0, the result is 0.
                if (num >> bit_pos) & 1:
                    count += 1
                    
            # If the count of numbers with the bit set is at least k,
            # then this bit should be set in the K-or result.
            if count >= k:
                # Set the bit at bit_pos in k_or_result.
                # We use bitwise OR with a number that has only the bit at bit_pos set.
                # (1 << bit_pos) creates such a number (which is equal to 2^bit_pos).
                k_or_result |= (1 << bit_pos)
                
        return k_or_result