from typing import List
import functools

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        # Convert each number to its binary string representation, removing the "0b" prefix.
        # Since 1 <= nums[i], bin(x)[2:] will not be "0" or "".
        binary_strings = [bin(x)[2:] for x in nums]

        # Custom comparison logic for sorting:
        # To form the largest number, we want the string that results in a lexicographically
        # larger concatenated string when placed earlier. For any two strings s1 and s2,
        # if concatenating s1 then s2 results in a larger number than concatenating s2 then s1,
        # then s1 should come before s2 in our desired sorted order.
        # We use functools.cmp_to_key to convert a comparison function (returning <0, 0, or >0)
        # into a key function compatible with Python's sorted().
        # The comparison function compare_func(s1, s2) should return:
        # - Negative if s1 is considered less than s2 (s1 comes before s2 in sorted list)
        # - Positive if s1 is considered greater than s2 (s1 comes after s2 in sorted list)
        # - Zero if s1 is considered equal to s2
        # We want s1 before s2 if int(s1 + s2, 2) > int(s2 + s1, 2).
        # So, compare_func(s1, s2) should return a negative value if int(s1 + s2, 2) > int(s2 + s1, 2).

        def compare_func(s1, s2):
            # Compare the numbers formed by concatenating s1+s2 and s2+s1
            val1 = int(s1 + s2, 2)
            val2 = int(s2 + s1, 2)

            # If val1 > val2, s1+s2 forms a larger number than s2+s1.
            # This means s1 should come before s2 in the final concatenation.
            # For standard ascending sort, s1 must be considered "smaller" than s2.
            if val1 > val2:
                return -1 # s1 is "smaller" (should come first)
            elif val1 < val2:
                return 1  # s1 is "larger" (should come later)
            else:
                return 0  # s1 and s2 are equivalent for sorting purposes

        # Sort the binary strings using the custom comparison function.
        # The sorted list will have the strings in the order that maximizes the concatenated binary number.
        sorted_binary_strings = sorted(binary_strings, key=functools.cmp_to_key(compare_func))

        # Join the sorted binary strings to get the final concatenated binary representation.
        concatenated_binary = "".join(sorted_binary_strings)

        # Convert the concatenated binary string to an integer (base 2).
        # The string is guaranteed not to be empty since nums has length 3 and nums[i] >= 1.
        # The string is also guaranteed not to have a leading '0' because nums[i] >= 1
        # means each binary string starts with '1', and the sort ensures a string
        # starting with '1' is placed first (unless all were '0', which is not the case).
        result = int(concatenated_binary, 2)

        return result