import itertools
from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        """
        Calculates the maximum number formed by concatenating binary representations
        of numbers in nums in some order.

        The problem asks for the largest integer value that can be obtained by:
        1. Taking the three numbers from the input list `nums`.
        2. Converting each number to its binary representation (without leading zeros, 
           which is standard for positive integers; the `bin()` function in Python 
           produces "0b..." prefix which needs removal).
        3. Concatenating these binary strings in some order (permutation).
        4. Converting the final concatenated binary string back to an integer.

        Since the input list `nums` always has a length of 3, there are 3! = 6 possible 
        orderings (permutations) of the numbers. We can simply try all 6 permutations, 
        calculate the resulting integer for each, and return the maximum value found.

        Args:
            nums: A list of three integers. Constraints guarantee nums.length == 3 
                  and 1 <= nums[i] <= 127 for each element.

        Returns:
            The maximum integer value achievable by concatenating the binary 
            representations of the numbers in nums in some permutation.
        """
        
        # Initialize a variable to store the maximum value found so far.
        # Start with 0, as all numbers are positive, the final results will be > 0.
        max_val = 0
        
        # Use itertools.permutations to generate all possible orderings of the elements in nums.
        # For nums = [a, b, c], this will yield tuples like (a, b, c), (a, c, b), etc.
        for p in itertools.permutations(nums):
            # p is a tuple representing one specific order, e.g., (nums[0], nums[1], nums[2])
            
            # Initialize an empty string to build the concatenated binary representation for this permutation.
            concatenated_binary_string = ""
            
            # Iterate through the numbers in the current permutation order.
            for num in p:
                # Convert the current number `num` to its binary representation string.
                # The `bin()` function returns a string prefixed with "0b" (e.g., bin(5) -> "0b101").
                # We slice the string starting from index 2 (`[2:]`) to remove this prefix.
                # Since nums[i] >= 1, the binary representation will not have leading zeros after removing "0b".
                binary_repr = bin(num)[2:]
                
                # Append this binary string to our accumulator string.
                concatenated_binary_string += binary_repr
            
            # After concatenating binary strings for all numbers in the current order `p`,
            # convert the resulting binary string back to an integer using base 2.
            # Example: If concatenated_binary_string is "11110", int("11110", 2) yields 30.
            current_value = int(concatenated_binary_string, 2)
            
            # Compare the integer value obtained from this permutation (`current_value`)
            # with the maximum value found so far (`max_val`). Update `max_val` if
            # `current_value` is greater.
            max_val = max(max_val, current_value)
            
        # After iterating through all 6 permutations, `max_val` will hold the maximum
        # integer value achievable according to the problem description. Return this value.
        return max_val