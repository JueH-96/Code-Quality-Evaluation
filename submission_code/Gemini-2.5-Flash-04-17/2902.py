from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        """
        Finds the maximum sum of a pair of numbers from nums such that the maximum
        digit in both numbers are equal.

        Args:
            nums: A list of integers.

        Returns:
            The maximum sum of such a pair, or -1 if no such pair exists.
        """

        def get_max_digit(n: int) -> int:
            """Helper function to find the maximum digit in an integer."""
            max_d = 0
            # Since constraints are 1 <= nums[i], n will always be >= 1.
            # No need to handle n=0 explicitly based on constraints.
            while n > 0:
                max_d = max(max_d, n % 10)
                n //= 10
            return max_d

        # Dictionary to store up to the two largest numbers encountered so far
        # for each maximum digit (0-9).
        # Keys are max digits (0-9), values are lists of numbers.
        # We only need to store the largest two for each digit to find the max sum pair.
        max_nums_by_digit = {}

        for num in nums:
            max_d = get_max_digit(num)

            if max_d not in max_nums_by_digit:
                max_nums_by_digit[max_d] = []

            # Add the current number to the list for its max digit
            current_list = max_nums_by_digit[max_d]
            current_list.append(num)

            # Keep the list sorted in descending order and trim to size 2
            # This ensures the list always contains the largest and second largest
            # numbers seen so far for this maximum digit.
            current_list.sort(reverse=True)
            if len(current_list) > 2:
                current_list.pop() # Remove the smallest (last one)

        max_overall_sum = -1

        # Iterate through the stored lists and calculate sums for pairs
        for numbers_list in max_nums_by_digit.values():
            # A pair can only be formed if there are at least two numbers
            # with the same maximum digit.
            if len(numbers_list) == 2:
                # The list is sorted descending, so the first two elements
                # are the largest and second largest numbers for this max digit.
                current_pair_sum = numbers_list[0] + numbers_list[1]
                max_overall_sum = max(max_overall_sum, current_pair_sum)

        return max_overall_sum