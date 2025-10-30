import math
from typing import List

class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of elements in nums whose corresponding indices 
        have exactly k set bits in their binary representation.

        Args:
            nums: A 0-indexed integer array.
            k: The target number of set bits for the index.

        Returns:
            An integer denoting the sum.
        """
        
        total_sum = 0
        n = len(nums)
        
        for i in range(n):
            # Calculate the number of set bits (1s) in the binary representation of the index i
            
            # Method 1: Using built-in bin() and count()
            # set_bits_count = bin(i).count('1') 
            
            # Method 2: Manual bit counting (Brian Kernighan's algorithm - efficient)
            temp_i = i
            set_bits_count = 0
            while temp_i > 0:
                temp_i &= (temp_i - 1) # Clears the least significant set bit
                set_bits_count += 1
                
            # Method 3: Using int.bit_count() (Python 3.10+)
            # if hasattr(int, 'bit_count'): # Check if method exists for compatibility
            #     set_bits_count = i.bit_count()
            # else: 
            #     # Fallback to another method if bit_count is not available
            #     set_bits_count = bin(i).count('1')

            # Check if the number of set bits equals k
            if set_bits_count == k:
                total_sum += nums[i]
                
        return total_sum

# Example usage (for testing outside the class structure if needed):
# sol = Solution()
# print(sol.sumIndicesWithKSetBits([5,10,1,5,2], 1)) # Output: 13
# print(sol.sumIndicesWithKSetBits([4,3,2,1], 2))    # Output: 1