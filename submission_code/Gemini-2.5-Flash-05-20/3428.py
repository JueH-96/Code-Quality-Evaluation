import collections
from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        """
        Calculates the bitwise XOR of all numbers that appear twice in the array.

        Args:
            nums: A list of integers where each number appears either once or twice.

        Returns:
            The bitwise XOR of all numbers that appear twice, or 0 if no number
            appears twice.
        """
        # Use collections.Counter to efficiently count the occurrences of each number.
        # This creates a dictionary-like object where keys are numbers from `nums`
        # and values are their respective frequencies.
        counts = collections.Counter(nums)
        
        # Initialize the result variable for the bitwise XOR sum.
        # Starting with 0 ensures that if no numbers appear twice, 0 is returned,
        # and XORing any number with 0 results in the number itself, making it
        # a neutral starting point for accumulation.
        result_xor = 0
        
        # Iterate through the items (number and its count) in the frequency map.
        for num, count in counts.items():
            # Check if the current number appeared exactly twice.
            if count == 2:
                # If it did, perform a bitwise XOR operation with the current result_xor.
                # This accumulates the XOR sum of all numbers found to be duplicates.
                result_xor ^= num
                
        # After checking all numbers, return the final accumulated XOR sum.
        return result_xor