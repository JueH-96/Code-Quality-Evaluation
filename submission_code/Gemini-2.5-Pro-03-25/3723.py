import math
from typing import List

class Solution:
    """
    Given an array of integers nums and an integer k, an element nums[i] is considered good
    if it is strictly greater than the elements at indices i - k and i + k (if those indices exist).
    If neither of these indices exists, nums[i] is still considered good.
    Return the sum of all the good elements in the array.
    """
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of good numbers in the array based on the given condition.

        An element nums[i] is good if it satisfies two conditions:
        1. It's strictly greater than nums[i - k], OR the index i - k is out of bounds (< 0).
        2. It's strictly greater than nums[i + k], OR the index i + k is out of bounds (>= n).

        Args:
            nums: A list of integers. The input array.
            k: An integer specifying the distance to check neighbors.

        Returns:
            The sum of all elements in nums that are considered "good".
        """
        n = len(nums)
        total_sum = 0

        # Iterate through each element in the array
        for i in range(n):
            current_num = nums[i]

            # Check the condition related to the element at index i - k
            # The condition is met if the index i - k is out of bounds (i.e., i - k < 0)
            # or if the index is valid and nums[i] is strictly greater than nums[i - k].
            # We use short-circuiting 'or' which avoids index out of bounds error.
            # If i - k < 0, the first part is True, and the second part isn't evaluated.
            # If i - k >= 0, the first part is False, and the second part is evaluated.
            left_condition_met = (i - k < 0) or (current_num > nums[i - k])

            # Check the condition related to the element at index i + k
            # The condition is met if the index i + k is out of bounds (i.e., i + k >= n)
            # or if the index is valid and nums[i] is strictly greater than nums[i + k].
            # We use short-circuiting 'or' similarly.
            # If i + k >= n, the first part is True, and the second part isn't evaluated.
            # If i + k < n, the first part is False, and the second part is evaluated.
            right_condition_met = (i + k >= n) or (current_num > nums[i + k])

            # An element nums[i] is considered "good" if both conditions are met.
            if left_condition_met and right_condition_met:
                total_sum += current_num

        return total_sum