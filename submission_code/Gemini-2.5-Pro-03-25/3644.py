import math
from typing import List

class Solution:
    """
    Finds the minimum positive sum of a subarray whose length is between l and r (inclusive).
    The approach uses prefix sums to calculate subarray sums efficiently in O(1) time
    after an initial O(N) preprocessing step. The overall time complexity is O(N^2)
    due to iterating through all relevant subarray start and end points.
    The space complexity is O(N) for storing the prefix sums.
    Given the constraint N <= 100, O(N^2) is acceptable.
    """
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        """
        Calculates the minimum positive sum of a subarray with length constraints.

        Args:
            nums: The input list of integers.
            l: The minimum length of the subarray.
            r: The maximum length of the subarray.

        Returns:
            The minimum positive sum found, or -1 if no such subarray exists.
        """
        n = len(nums)
        
        # Calculate prefix sums
        # prefix_sum[k] stores the sum of the first k elements (nums[0]...nums[k-1]).
        # prefix_sum[0] = 0
        # prefix_sum[k] = nums[0] + ... + nums[k-1] for k > 0
        # The size of the prefix_sum array is n + 1.
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
            
        # Initialize the minimum positive sum found so far to positive infinity.
        # This value will be updated if a valid positive sum is found.
        min_positive_sum = float('inf')
        
        # Iterate through all possible ending positions 'k' (using prefix sum indices).
        # 'k' represents the index in the prefix_sum array, which is one position 
        # after the actual end index of the subarray in the original 'nums' array.
        # A subarray nums[i...j] corresponds to prefix_sum indices i and k=j+1.
        # The length of the subarray nums[i...k-1] is k - i.
        # The minimum possible length is l, so the smallest k to consider is l 
        # (corresponding to subarray nums[0...l-1]).
        # The maximum possible length is constrained by n, so the largest k is n
        # (corresponding to subarrays ending at index n-1).
        for k in range(l, n + 1):
            # For a fixed ending position 'k', determine the range of possible starting 
            # positions 'i' (using prefix sum indices).
            # 'i' corresponds to the actual starting index of the subarray in 'nums'.
            # The length constraint is l <= k - i <= r.
            # From k - i <= r, we get i >= k - r.
            # From k - i >= l, we get i <= k - l.
            # Also, the starting index 'i' must be non-negative, so i >= 0.
            
            # Combine the constraints to find the valid range for the starting index 'i':
            # max(0, k - r) <= i <= k - l.
            start_i = max(0, k - r)
            end_i = k - l
            
            # Iterate through all valid starting indices 'i' for the current ending position 'k'.
            # The range function handles the case where start_i > end_i by yielding no values.
            for i in range(start_i, end_i + 1):
                # Calculate the sum of the subarray nums[i...k-1] using prefix sums.
                # The sum is prefix_sum[k] - prefix_sum[i].
                current_sum = prefix_sum[k] - prefix_sum[i]
                
                # Check if the calculated sum is positive (> 0).
                if current_sum > 0:
                    # If the sum is positive, update the overall minimum positive sum found so far.
                    min_positive_sum = min(min_positive_sum, current_sum)
                    
        # After iterating through all possible subarrays satisfying the length constraints:
        # Check if min_positive_sum was updated from its initial infinite value.
        # If it remains infinity, it means no subarray met the conditions 
        # (length between l and r, and sum > 0).
        if min_positive_sum == float('inf'):
            # Return -1 if no valid subarray was found.
            return -1
        else:
            # Otherwise, return the minimum positive sum found.
            # Since all input numbers and sums are integers, the result is an integer.
            return min_positive_sum