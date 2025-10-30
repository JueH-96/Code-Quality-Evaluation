from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        """
        Counts the number of subarrays of length 3 where the sum of the first and third numbers
        equals exactly half of the second number.

        Args:
            nums: An integer array.

        Returns:
            The number of such subarrays satisfying the condition.
        """
        count = 0
        n = len(nums)

        # We need to consider all subarrays of length 3.
        # A subarray of length 3 starting at index i includes elements at indices i, i+1, and i+2.
        # Since the length of the array is n, the elements are at indices 0 to n-1.
        # The last possible starting index i for a subarray of length 3 is when the third element
        # is at the last index of the array (n-1). So, i + 2 = n - 1, which means i = n - 3.
        # The starting index i can range from 0 up to n-3 (inclusive).
        # The range(n - 2) function generates indices from 0 up to (n - 2) - 1 = n - 3.
        for i in range(n - 2):
            # Get the three elements of the current subarray of length 3
            first = nums[i]
            second = nums[i+1]
            third = nums[i+2]

            # Check the condition: sum of the first and third numbers equals exactly half of the second number.
            # first + third == second / 2
            # Since all numbers are integers, for this equality to hold, 'second' must be even,
            # and 'first + third' must equal the integer result of 'second / 2'.
            # An equivalent check using integer arithmetic is: (first + third) * 2 == second.
            # This avoids potential floating-point issues and correctly handles the implicit requirement
            # that 'second' must be divisible by 2 such that its half is an integer equal to first + third.
            if (first + third) * 2 == second:
                count += 1

        return count