import math # math module is not strictly needed as float('-inf') is used but keeping it doesn't hurt. It can be removed.
from typing import List

class Solution:
    """
    Solves the problem of finding the maximum number of distinct elements
    after applying an operation on each element at most once.
    The operation allows adding an integer in [-k, k] to the element.
    """
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        """
        Calculates the maximum possible number of distinct elements using a greedy approach.

        The core idea is to sort the array and then iterate through it. For each element,
        we try to assign it the smallest possible value that is distinct from the previously
        assigned values and falls within the allowed range [nums[i] - k, nums[i] + k].
        By choosing the smallest possible value, we maximize the chances for subsequent elements
        to also find distinct values.

        Args:
            nums: A list of integers.
            k: An integer representing the range of the operation [-k, k].

        Returns:
            The maximum number of distinct elements achievable.
        """
        
        # Sort the array. This allows us to process elements greedily.
        # By processing smaller numbers first, we can try to assign them the smallest
        # possible distinct values, leaving larger values available for later elements.
        nums.sort()
        
        distinct_count = 0
        # Initialize last_assigned to negative infinity. This represents that no value
        # has been assigned yet. Using float('-inf') ensures that the condition
        # `potential_next_val > last_assigned` is effectively handled for the first assignment.
        # Any valid assigned value `v` will satisfy `v > float('-inf')`.
        last_assigned = float('-inf') 
        
        n = len(nums)
        # Iterate through each number in the sorted array.
        for i in range(n):
            current_num = nums[i]
            
            # Calculate the target value we want to assign for the current element nums[i].
            # This target value must satisfy two conditions:
            # 1. It must be strictly greater than the last assigned value to ensure distinctness.
            #    So, it must be at least `last_assigned + 1`.
            # 2. The operation restricts the assigned value to be at least `current_num - k`.
            # We take the maximum of these two lower bounds to find the smallest possible 
            # candidate value (`potential_next_val`) that satisfies both lower bound constraints.
            potential_next_val = max(current_num - k, last_assigned + 1)
            
            # Check if this smallest possible candidate value is achievable within the 
            # allowed operation range for `current_num`. The operation restricts the assigned
            # value to be at most `current_num + k`.
            # So, we need to check if `potential_next_val <= current_num + k`.
            # Note: We already ensured `potential_next_val >= current_num - k` by the `max` operation above.
            if potential_next_val <= current_num + k:
                # If `potential_next_val` is within the allowed range [current_num - k, current_num + k],
                # it means we can assign this value to `nums[i]` (conceptually).
                # We increment the count of distinct elements found.
                distinct_count += 1
                # Update the `last_assigned` value to `potential_next_val`. This ensures
                # the next assigned value must be strictly greater than this one.
                last_assigned = potential_next_val
            # Else (if potential_next_val > current_num + k): 
            # This means even the smallest candidate value that meets the distinctness requirement 
            # (`> last_assigned`) and the operation's lower bound (`>= current_num - k`) 
            # is greater than the operation's upper bound (`current_num + k`).
            # Therefore, we cannot use the current element `nums[i]` to form a new distinct element 
            # greater than `last_assigned`. We skip this element and proceed to the next one, 
            # leaving `last_assigned` unchanged.
            
        # Return the total count of distinct elements we were able to assign.
        return distinct_count