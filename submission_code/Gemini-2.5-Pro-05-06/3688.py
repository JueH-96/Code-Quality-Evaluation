import collections
from typing import List

class Solution:
    def _kadane(self, arr: List[int]) -> int:
        # Assumes arr is non-empty, which is guaranteed by the problem logic.
        # If arr were empty, a max non-empty subarray sum is undefined.
        # One might return -float('inf') or raise an error.
        # Given constraints, arr fed to _kadane will always be non-empty.
        
        max_so_far = arr[0]
        current_max_ending_here = arr[0]
        for i in range(1, len(arr)):
            num = arr[i]
            current_max_ending_here = max(num, current_max_ending_here + num)
            max_so_far = max(max_so_far, current_max_ending_here)
        return max_so_far

    def maxSubarraySum(self, nums: List[int]) -> int:
        if not nums:
            # Problem constraint says 1 <= nums.length, so this won't be hit.
            # If it could, proper return value depends on definition for empty array.
            return 0 # Or an error, or -float('inf')

        # Calculate max subarray sum for the original array
        max_overall_sum = self._kadane(nums)

        # Get unique values and their counts
        counts = collections.Counter(nums)
        
        # We only need to consider removing x < 0.
        # Proof: If x >= 0 is removed, max subarray sum won't increase.
        # Let S0 = _kadane(nums). Let A' = nums_without_x. S' = _kadane(A').
        # Let B' be the subarray of A' summing to S'.
        # Let B_orig be the segment in nums spanning elements of B'.
        # sum(B_orig) = S' + (count_x_in_B_orig * x). Since x >= 0, sum(B_orig) >= S'.
        # S0 >= sum(B_orig), so S0 >= S'.
        
        unique_negative_values = [val for val in counts if val < 0]

        for x_to_remove in unique_negative_values:
            # Check if removing x_to_remove would make the array empty.
            # This happens if all elements in nums are x_to_remove.
            if counts[x_to_remove] == len(nums):
                # This removal is not allowed by problem spec ("nums remains non-empty")
                continue

            # Create the temporary array by removing all occurrences of x_to_remove
            # This list comprehension is O(N)
            temp_nums = [num for num in nums if num != x_to_remove]
            
            # temp_nums is guaranteed to be non-empty here because:
            # len(nums) >= 1 (problem constraint)
            # counts[x_to_remove] < len(nums) (checked above)
            # So, there's at least one element in nums that is not x_to_remove.
            
            current_max_sum_for_temp_array = self._kadane(temp_nums)
            if current_max_sum_for_temp_array > max_overall_sum:
                max_overall_sum = current_max_sum_for_temp_array
                
        return max_overall_sum