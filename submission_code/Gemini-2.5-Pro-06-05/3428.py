from collections import Counter
from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        """
        Calculates the bitwise XOR of all numbers that appear twice in the array.
        """
        
        # Use collections.Counter to efficiently count the occurrences of each number.
        # This creates a frequency map, e.g., {1: 2, 2: 1, 3: 1} for nums = [1,2,1,3].
        counts = Counter(nums)
        
        # Initialize a variable to store the bitwise XOR sum.
        # The identity element for XOR is 0. If no numbers appear twice,
        # this initial value of 0 will be the final correct result.
        xor_sum = 0
        
        # Iterate over the unique numbers and their corresponding frequencies.
        for num, frequency in counts.items():
            # We are looking for numbers that appear exactly twice.
            if frequency == 2:
                # If a number appears twice, update the total XOR sum.
                xor_sum ^= num
                
        return xor_sum