import math # Not used, but standard practice to import modules if potentially needed. Can be removed.
from typing import List

class Solution:
    """
    Calculates the minimum number of operations to make the median of nums equal to k.

    An operation consists of increasing or decreasing any element by 1.
    The median is the middle element after sorting, or the larger of the two middle
    elements if the array length is even (element at index n // 2 in 0-based indexing).
    """
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        """
        Args:
            nums: A list of integers.
            k: The target median value (non-negative integer).

        Returns:
            The minimum number of operations required.
        """

        # Sort the input array to easily identify the median and elements relative to it.
        # The sorting step takes O(n log n) time complexity.
        nums.sort()

        # Get the length of the array.
        n = len(nums)

        # Determine the index of the median element according to the problem definition.
        # If n is odd (e.g., 5), median index is n // 2 = 2 (element at index 2).
        # If n is even (e.g., 6), median index is n // 2 = 3 (element at index 3, the larger of the two middle).
        # In 0-based indexing, this is always the element at index n // 2.
        median_index = n // 2

        # Initialize the total number of operations needed.
        # Python integers handle arbitrary size, so overflow is not an issue for the given constraints.
        operations: int = 0

        # Calculate the minimum operations required based on the sorted array and target median k.
        # The core idea is that to minimize operations, we only need to adjust elements
        # that violate the final sorted order with k at the median position.
        # Specifically:
        # 1. The element at the median index `median_index` must become `k`.
        # 2. All elements at indices `i <= median_index` must be less than or equal to `k`.
        # 3. All elements at indices `i >= median_index` must be greater than or equal to `k`.
        #
        # The minimum cost to satisfy these conditions can be calculated efficiently.
        # We derived the total cost formula as:
        # Cost = sum_{i=0}^{median_index} max(0, nums[i] - k) + sum_{i=median_index}^{n-1} max(0, k - nums[i])
        # This formula correctly accounts for the cost of adjusting the median element itself
        # and all other elements as necessary.

        # Calculate the first part of the sum: operations for elements from index 0 up to median_index.
        # If nums[i] > k, it must be decreased to k, requiring nums[i] - k operations.
        # If nums[i] <= k, no operation is needed for this element based on the condition `y_i <= k`.
        # The cost for element i in this range is max(0, nums[i] - k).
        for i in range(median_index + 1):
             operations += max(0, nums[i] - k)

        # Calculate the second part of the sum: operations for elements from median_index to the end.
        # If nums[i] < k, it must be increased to k, requiring k - nums[i] operations.
        # If nums[i] >= k, no operation is needed for this element based on the condition `y_i >= k`.
        # The cost for element i in this range is max(0, k - nums[i]).
        # Note: The median element at `median_index` is considered in both loops.
        # The sum `max(0, nums[median_index] - k) + max(0, k - nums[median_index])`
        # correctly equals `abs(nums[median_index] - k)`, which is the cost to adjust the median element itself.
        for i in range(median_index, n):
             operations += max(0, k - nums[i])

        # Return the total calculated minimum number of operations.
        # The overall time complexity is dominated by sorting: O(n log n).
        # The space complexity depends on the sorting algorithm used (e.g., O(n) for Timsort in Python, or O(log n) for heapsort).
        return operations