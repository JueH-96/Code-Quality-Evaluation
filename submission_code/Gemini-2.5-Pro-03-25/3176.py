import math
from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        """
        Calculates the minimum mountain triplet sum.

        A mountain triplet (i, j, k) satisfies:
        1. i < j < k
        2. nums[i] < nums[j] and nums[k] < nums[j]

        Args:
            nums: A 0-indexed list of integers. Length is between 3 and 50. Values are between 1 and 50.

        Returns:
            The minimum sum of a mountain triplet, or -1 if no such triplet exists.
        
        The algorithm uses precomputed prefix and suffix minimums to efficiently find the 
        minimum possible left and right elements for each potential peak element nums[j]. 
        This approach achieves O(n) time complexity and O(n) space complexity.
        """
        n = len(nums)
        
        # Constraints state 3 <= n <= 50, so the array has at least 3 elements.

        # Calculate prefix minimums: prefix_min[j] = min(nums[0...j-1])
        # prefix_min[j] stores the minimum value encountered in the array strictly *before* index j.
        # Initialize with infinity, as there's no element before index 0.
        prefix_min = [math.inf] * n 
        # Keep track of the minimum value found so far as we iterate from left to right.
        current_min = math.inf
        for j in range(n):
            # Before updating current_min with nums[j], store the minimum found up to j-1.
            # This is only relevant for j > 0. prefix_min[0] remains inf.
            if j > 0:
                prefix_min[j] = current_min 
            # Update the overall minimum seen so far, including the current element nums[j].
            current_min = min(current_min, nums[j])

        # Calculate suffix minimums: suffix_min[j] = min(nums[j+1...n-1])
        # suffix_min[j] stores the minimum value encountered in the array strictly *after* index j.
        # Initialize with infinity, as there's no element after index n-1.
        suffix_min = [math.inf] * n
        # Keep track of the minimum value found so far as we iterate from right to left.
        current_min = math.inf
        for j in range(n - 1, -1, -1):
             # Before updating current_min with nums[j], store the minimum found from j+1 onwards.
             # This is only relevant for j < n-1. suffix_min[n-1] remains inf.
            if j < n - 1:
                suffix_min[j] = current_min
             # Update the overall minimum seen so far (from the right), including the current element nums[j].
            current_min = min(current_min, nums[j])

        # Initialize the minimum sum found so far to infinity.
        min_total_sum = math.inf

        # Iterate through each element nums[j] considering it as the potential peak (middle element) of a mountain triplet.
        # The peak index j must be between 1 and n-2 (inclusive) to allow for elements to its left (index i < j) and right (index k > j).
        for j in range(1, n - 1):
            left_min = prefix_min[j]  # The minimum value among elements nums[0...j-1]
            right_min = suffix_min[j] # The minimum value among elements nums[j+1...n-1]
            current_num = nums[j]     # The value at the potential peak index j.

            # Check if nums[j] can form a peak of a mountain triplet.
            # This requires:
            # 1. There exists an element to its left that is smaller (guaranteed if left_min < current_num).
            # 2. There exists an element to its right that is smaller (guaranteed if right_min < current_num).
            if left_min < current_num and right_min < current_num:
                # If both conditions are true, a mountain triplet centered at j exists.
                # The sum using the minimum possible left value (left_min) and minimum possible 
                # right value (right_min) constitutes a valid mountain triplet sum.
                # We calculate this sum as a candidate for the overall minimum sum.
                current_sum = left_min + current_num + right_min
                # Update the overall minimum sum found across all possible peaks j.
                min_total_sum = min(min_total_sum, current_sum)

        # After checking all potential peaks j:
        # If min_total_sum is still infinity, it means no triplet satisfied the mountain conditions.
        if min_total_sum == math.inf:
            return -1
        else:
            # Otherwise, return the minimum sum found.
            # Since all input numbers and sums are integers, the result is an integer.
            return min_total_sum