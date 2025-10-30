import itertools
from typing import List # Required for the type hint List[int]

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        # Step 1: Convert each number in nums to its binary string representation.
        # bin(num)[2:] removes the "0b" prefix.
        # E.g., bin(3) is "0b11", so bin(3)[2:] is "11".
        # Constraints 1 <= nums[i] <= 127 ensure nums[i] > 0.
        # The problem note "binary representation of any number does not contain leading zeros"
        # is naturally handled by bin(num)[2:] for positive integers.
        binary_strings = [bin(num)[2:] for num in nums]

        max_numeric_value = 0  # Initialize with 0, as results are positive.

        # Step 2: Generate all permutations of these binary strings.
        # For 3 elements, there are 3! = 6 permutations.
        # itertools.permutations returns an iterator of tuples.
        # Each tuple 'p_tuple' contains the binary strings in a specific order.
        for p_tuple in itertools.permutations(binary_strings):
            # Step 3a: Concatenate the binary strings in the current permuted order.
            # E.g., if p_tuple is ("11", "1", "10"), this forms "11110".
            concatenated_binary_string = "".join(p_tuple)
            
            # Step 3b: Convert the concatenated binary string to an integer.
            # int(string, 2) converts a base-2 string to its decimal equivalent.
            # E.g., int("11110", 2) becomes 30.
            current_numeric_value = int(concatenated_binary_string, 2)
            
            # Step 4: Keep track of the maximum integer value found so far.
            max_numeric_value = max(max_numeric_value, current_numeric_value)
                
        # Step 5: Return the overall maximum value.
        return max_numeric_value