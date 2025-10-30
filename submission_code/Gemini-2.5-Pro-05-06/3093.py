from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        total_sum = 0  # Initialize the sum
        
        # Iterate through the indices of the nums array
        for i in range(len(nums)):
            # Count the number of set bits (1s) in the binary representation of index i.
            # bin(i) converts an integer to its binary string, e.g., bin(5) -> "0b101".
            # .count('1') then counts the '1' characters in that string.
            # For i=0, bin(0) is "0b0", so bin(0).count('1') is 0.
            set_bits_in_index = bin(i).count('1')
            
            # If the number of set bits in the current index i is equal to k
            if set_bits_in_index == k:
                # Add the element at this index to our total sum
                total_sum += nums[i]
                
        return total_sum