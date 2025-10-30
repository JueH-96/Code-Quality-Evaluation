from typing import List

class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)  # Get the length of the array, denoted as n
        sum_of_squares = 0  # Initialize the sum of squares

        # The problem defines 'i' as a 1-based index.
        # We iterate 'i' from 1 to n (inclusive).
        for i in range(1, n + 1):
            # An element nums[i] (using 1-based indexing as per problem statement)
            # is special if its index 'i' divides 'n'.
            if n % i == 0:
                # If 'i' divides 'n', then the element is special.
                # Python lists are 0-indexed. So, the element at 1-based index 'i'
                # is located at `nums[i-1]` in the actual list.
                special_element_value = nums[i-1]
                
                # Add the square of this special element's value to our sum.
                sum_of_squares += special_element_value * special_element_value
                
        return sum_of_squares