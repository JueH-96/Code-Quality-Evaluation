import collections
from typing import List

class Solution:
    def _get_max_digit(self, n: int) -> int:
        """
        Helper function to find the maximum digit in an integer.
        Converts the number to a string and then finds the maximum character
        (which represents the largest digit) and converts it back to an integer.
        
        Example:
        _get_max_digit(71)   -> max('7', '1') is '7' -> 7
        _get_max_digit(104)  -> max('1', '0', '4') is '4' -> 4
        _get_max_digit(9)    -> max('9') is '9' -> 9
        """
        return int(max(str(n)))

    def maxSum(self, nums: List[int]) -> int:
        # A dictionary to group numbers by their maximum digit.
        # keys: maximum digit (0-9)
        # values: a list of numbers from `nums` that have that maximum digit.
        # Using collections.defaultdict(list) simplifies adding elements to lists
        # without checking if the key already exists.
        max_digit_groups = collections.defaultdict(list)

        # Step 1 & 2: Populate the max_digit_groups dictionary by finding
        # the maximum digit for each number and grouping them.
        for num in nums:
            max_digit = self._get_max_digit(num)
            max_digit_groups[max_digit].append(num)
        
        # Initialize the maximum sum found so far to -1.
        # This handles the case where no such pair exists, as required by the problem.
        overall_max_sum = -1

        # Step 3: Iterate through each group of numbers (grouped by their max digit).
        # We iterate over the values (lists of numbers) in the dictionary.
        for digit_group in max_digit_groups.values():
            # A pair can only be formed if there are at least two numbers in the group.
            if len(digit_group) >= 2:
                # To find the maximum sum of a pair within this group, we need
                # the two largest numbers. Sorting the group in descending order
                # allows us to easily pick the top two numbers from indices 0 and 1.
                # Given nums.length <= 100, sorting a sublist of up to 100 elements
                # is computationally efficient enough.
                digit_group.sort(reverse=True)
                
                # Calculate the sum of the two largest numbers in this group.
                current_pair_sum = digit_group[0] + digit_group[1]
                
                # Update the overall maximum sum if the current pair's sum is greater.
                overall_max_sum = max(overall_max_sum, current_pair_sum)
                
        # Step 4: Return the maximum sum found or -1 if no valid pair was found.
        return overall_max_sum