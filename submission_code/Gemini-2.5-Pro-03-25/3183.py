import math
from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        """
        Calculates the K-or of a list of integers.

        The i-th bit is set in the K-or if and only if there are at least
        k elements of nums in which bit i is set.

        Args:
            nums: A list of 0-indexed non-negative integers.
            k: An integer threshold.

        Returns:
            The K-or of the nums array.
        """
        
        k_or_result = 0
        
        # We need to check bits from 0 up to the maximum possible bit set in any number.
        # Since nums[i] < 2^31, the maximum bit position we need to consider is 30.
        # We iterate through each bit position i from 0 to 30.
        for i in range(31):
            count = 0  # Counter for numbers with the i-th bit set
            
            # Calculate the value represented by the i-th bit (2^i)
            bit_value = 1 << i 
            
            # Iterate through each number in the input array
            for num in nums:
                # Check if the i-th bit is set in the current number 'num'
                # We can do this using the bitwise AND operator:
                # If (num AND bit_value) is non-zero, the i-th bit is set.
                if (num & bit_value) == bit_value:
                    count += 1
            
            # If the count of numbers with the i-th bit set is at least k
            if count >= k:
                # Set the i-th bit in the k_or_result
                # We use the bitwise OR operator to add the bit_value to the result
                k_or_result = k_or_result | bit_value
                # Alternatively: k_or_result += bit_value
                
        return k_or_result