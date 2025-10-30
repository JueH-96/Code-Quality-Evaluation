from typing import List
import collections # collections import is not strictly necessary for this solution, but good practice if using its types later

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        """
        Calculates the sum of the squares of distinct counts of all subarrays of nums.

        The distinct count of a subarray nums[i..j] is defined as the number of unique elements
        present in that subarray. This function iterates through all possible contiguous subarrays,
        computes the square of the distinct count for each, and sums these squares.

        Args:
            nums: A 0-indexed integer array. Constraints: 1 <= nums.length <= 100, 1 <= nums[i] <= 100.

        Returns:
            The sum of the squares of the distinct counts of all subarrays of nums.
        """
        n = len(nums)
        # Initialize the total sum which will store the sum of squares of distinct counts
        total_sum_of_squares = 0

        # Iterate through all possible starting indices 'i' of the subarrays.
        # 'i' ranges from 0 to n-1.
        for i in range(n):
            # Initialize a set to store distinct elements for subarrays starting at index 'i'.
            # A set is used because it efficiently stores unique elements and provides
            # O(1) average time complexity for adding elements and checking size.
            distinct_elements_in_subarray = set()

            # Iterate through all possible ending indices 'j' of the subarrays, starting from 'i'.
            # 'j' ranges from 'i' to n-1.
            # This ensures we consider all subarrays nums[i..j] where 0 <= i <= j < n.
            for j in range(i, n):
                # The current subarray under consideration is nums[i..j].

                # Add the element at index 'j' (the rightmost element of the expanding subarray)
                # to the set of distinct elements. If the element is already present, the set
                # remains unchanged due to the nature of sets.
                distinct_elements_in_subarray.add(nums[j])

                # The number of distinct elements in the subarray nums[i..j] is simply the
                # current size (cardinality) of the set.
                distinct_count = len(distinct_elements_in_subarray)

                # Square the distinct count and add it to the running total sum.
                # Using `distinct_count * distinct_count` is equivalent to `distinct_count ** 2`.
                total_sum_of_squares += distinct_count * distinct_count

        # After iterating through all possible subarrays (defined by start 'i' and end 'j'),
        # return the final computed sum of the squares of their distinct counts.
        return total_sum_of_squares