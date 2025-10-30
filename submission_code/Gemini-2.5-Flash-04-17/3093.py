from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of elements in nums whose corresponding indices 
        have exactly k set bits in their binary representation.

        Args:
            nums: A list of integers.
            k: An integer representing the required number of set bits.

        Returns:
            An integer denoting the sum of elements at relevant indices.
        """
        total_sum = 0
        
        # Iterate through the list using enumerate to get both index and value
        for i, num in enumerate(nums):
            # Count the number of set bits (1s) in the binary representation of the index i
            # bin(i) converts the integer i to its binary string representation (e.g., 5 -> '0b101')
            # .count('1') counts the occurrences of '1' in the binary string
            set_bits_count = bin(i).count('1')
            
            # Check if the number of set bits is exactly k
            if set_bits_count == k:
                # If it is, add the element num at this index to the total sum
                total_sum += num
        
        # Return the final calculated sum
        return total_sum