import math
from typing import List

class Solution:
    """
    Represents a solution to the problem of finding the minimum cost to divide an array into three subarrays.
    The cost of an array is defined as its first element.
    The goal is to divide the input array `nums` into three disjoint contiguous subarrays
    such that the sum of their costs (first elements) is minimized.
    """
    def minimumCost(self, nums: List[int]) -> int:
        """
        Calculates the minimum cost to divide the array nums into three disjoint contiguous subarrays.

        The division must result in exactly three non-empty subarrays that cover the original array.
        The cost of a subarray is its first element. The total cost is the sum of the costs
        of the three subarrays.

        Args:
            nums: A list of integers. The length n satisfies 3 <= n <= 50.
                  Each element nums[i] satisfies 1 <= nums[i] <= 50.

        Returns:
            The minimum possible sum of the costs of the three subarrays.

        Example:
            If nums = [1, 2, 3, 12], the possible divisions are:
            - [1], [2], [3, 12] -> cost = 1 + 2 + 3 = 6
            - [1], [2, 3], [12] -> cost = 1 + 2 + 12 = 15
            - [1, 2], [3], [12] -> cost = 1 + 3 + 12 = 16
            The minimum cost is 6.

        Explanation of the approach:
        Let the three contiguous subarrays be nums[0...i-1], nums[i...j-1], and nums[j...n-1].
        The indices must satisfy 1 <= i < j <= n-1 to ensure three non-empty subarrays.
        The costs of these subarrays are nums[0], nums[i], and nums[j], respectively.
        The total cost is nums[0] + nums[i] + nums[j].
        Since nums[0] is always part of the cost (as it's the first element of the first subarray),
        we need to minimize the sum nums[i] + nums[j] over all possible pairs (i, j) such that 1 <= i < j <= n-1.
        The minimum value of nums[i] + nums[j] occurs when nums[i] and nums[j] correspond to the
        two smallest elements in the subarray nums[1...n-1].
        Therefore, the algorithm finds the two smallest elements in nums[1:] and adds them to nums[0].
        """
        n = len(nums)

        # The first subarray always starts at index 0.
        # Its cost is fixed at nums[0].
        cost1 = nums[0]

        # We need to find the minimum sum of costs for the second and third subarrays.
        # This sum is determined by the two smallest elements in the rest of the array (nums[1:]).

        # Find the two smallest elements in the subarray nums[1:].
        # Initialize the smallest (min1) and second smallest (min2) values found so far.
        # Using math.inf ensures that the first two elements encountered will update these values.
        min1 = math.inf
        min2 = math.inf

        # Iterate through the elements from index 1 to n-1.
        # We are looking for the two smallest values in this range.
        for k in range(1, n):
            current_num = nums[k]

            if current_num <= min1:
                # If the current number is less than or equal to the current smallest (min1),
                # it becomes the new smallest. The old smallest (min1) becomes the
                # second smallest (min2). This handles cases where the minimum value
                # appears multiple times correctly.
                min2 = min1
                min1 = current_num
            elif current_num < min2:
                # If the current number is greater than min1 but less than the current
                # second smallest (min2), it becomes the new second smallest.
                min2 = current_num
            # If current_num >= min2, it doesn't affect the two smallest elements found so far.

        # The minimum possible sum for the costs of the second and third subarrays
        # is the sum of the two smallest elements found in nums[1:].
        min_sum_cost2_cost3 = min1 + min2

        # The total minimum cost is the sum of the cost of the first subarray (always nums[0])
        # and the minimum possible sum of costs for the second and third subarrays.
        total_min_cost = cost1 + min_sum_cost2_cost3

        return total_min_cost