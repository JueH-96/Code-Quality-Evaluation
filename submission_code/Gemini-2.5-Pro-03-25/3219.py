import collections
from typing import List # Make sure this is imported for type hinting

class Solution:
  """
  Finds the lexicographically smallest array obtainable by swapping elements 
  nums[i] and nums[j] if |nums[i] - nums[j]| <= limit.

  The core idea is that the ability to swap elements based on their value difference
  partitions the indices into several groups. Within each group, the elements
  originally at those indices can be arbitrarily permuted among themselves.
  To find these groups, we can sort the numbers along with their original indices.
  Then, iterate through the sorted list: elements with values `v1`, `v2` such that
  `|v1 - v2| <= limit` belong to the same group. This property is transitive.
  So, we can group adjacent elements in the sorted list if their difference is within `limit`.

  Once a group is identified, it consists of a set of values and a set of original indices.
  To obtain the lexicographically smallest array overall, we must arrange the values
  within each group such that the smallest values occupy the smallest indices within that group.
  This is achieved by sorting the values of the group (which are conveniently already sorted
  due to the initial sort of `indexed_nums`) and sorting the indices of the group, then
  assigning the k-th smallest value to the k-th smallest index.

  The final result array is constructed by performing this assignment for all groups.
  """
  def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
    """
    :param nums: A 0-indexed array of positive integers.
    :param limit: A positive integer limit for swapping condition.
    :return: The lexicographically smallest array obtainable.
    """
    n = len(nums)
    
    # Step 1: Create pairs of (value, original_index)
    # This tuple stores the value and its original position in the input array.
    # We need the original index to place the final value correctly in the result array.
    indexed_nums = []
    for i in range(n):
        indexed_nums.append((nums[i], i))
    
    # Step 2: Sort the pairs based on values
    # Sorting allows us to easily group elements with small differences.
    # The complexity of sorting is O(N log N).
    indexed_nums.sort() 
    
    # Step 3: Initialize the result array
    # We will fill this array as we process each group.
    result = [0] * n
    
    # Step 4: Iterate through sorted pairs to identify groups and process them
    # This loop iterates through all N elements once. The overall complexity inside
    # the loop except for sorting indices is O(N).
    i = 0
    while i < n:
        # A new group starts at index `i` in the sorted list `indexed_nums`.
        # Collect indices and values for the current group.
        current_group_indices = []
        current_group_values = []
        
        # The first element indexed_nums[i] is part of the new group.
        current_group_indices.append(indexed_nums[i][1]) # Add original index
        current_group_values.append(indexed_nums[i][0]) # Add value
        
        # Move to the next potential element of the group.
        i += 1
        
        # Extend the group with subsequent elements in the sorted list.
        # An element indexed_nums[i] is part of the current group if its value
        # differs from the previous element's value (indexed_nums[i-1]) by at most `limit`.
        while i < n and indexed_nums[i][0] - indexed_nums[i-1][0] <= limit:
            current_group_indices.append(indexed_nums[i][1])
            current_group_values.append(indexed_nums[i][0])
            i += 1
        
        # Step 5: Process the completed group
        # The `current_group_values` list contains the values of the group members.
        # Since we iterated through the sorted `indexed_nums`, these values are already sorted.
        
        # Sort the original indices associated with this group.
        # Let k be the size of the current group. Sorting takes O(k log k).
        # Sum of k log k over all groups is bounded by O(N log N).
        current_group_indices.sort()
        
        # Assign the sorted values to the sorted index positions in the result array.
        # This loop runs k times for a group of size k. Total time O(N).
        # For the lexicographically smallest result, the smallest value in the group
        # must go to the smallest index in the group, the second smallest value to the
        # second smallest index, and so on.
        for k in range(len(current_group_indices)):
            # Get the k-th smallest index from the sorted list of indices for this group.
            result_idx = current_group_indices[k]
            # Get the k-th smallest value from the sorted list of values for this group.
            value_to_assign = current_group_values[k]
            # Place the value into the result array at the determined index.
            result[result_idx] = value_to_assign
            
    # Step 6: Return the fully constructed result array
    # This array represents the lexicographically smallest permutation achievable
    # through the allowed swap operations.
    # Overall time complexity: O(N log N) dominated by the initial sorting.
    # Overall space complexity: O(N) for storing indexed_nums, result, and temporary group lists.
    return result