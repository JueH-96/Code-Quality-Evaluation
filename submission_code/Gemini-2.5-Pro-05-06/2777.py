from typing import List

class Solution:
  def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
    n = len(nums)
    
    # prefix_distinct_counts[i] will store the number of distinct elements 
    # in the prefix nums[0...i].
    prefix_distinct_counts = [0] * n
    current_prefix_elements = set()
    for i in range(n):
        current_prefix_elements.add(nums[i])
        prefix_distinct_counts[i] = len(current_prefix_elements)
        
    # The result array.
    ans = [0] * n
    
    # current_suffix_elements will store distinct elements encountered from the right.
    # When processing index `i` (iterating from right to left, i.e., i = n-1, n-2, ..., 0),
    # current_suffix_elements will hold the distinct elements from nums[i+1 ... n-1].
    current_suffix_elements = set()
    
    for i in range(n - 1, -1, -1): # Iterate i from n-1 down to 0
        # Number of distinct elements in prefix nums[0, ..., i].
        num_distinct_prefix = prefix_distinct_counts[i]
        
        # Number of distinct elements in suffix nums[i + 1, ..., n - 1].
        # For i = n-1 (the first iteration of this backward loop), 
        # current_suffix_elements is empty, so num_distinct_suffix is 0. This is correct as
        # the suffix nums[n ... n-1] is empty according to the problem note (i > j implies empty subarray).
        # For other i, current_suffix_elements contains distinct elements from
        # nums[i+1], nums[i+2], ..., nums[n-1] which were added in previous iterations of this loop.
        num_distinct_suffix = len(current_suffix_elements)
        
        ans[i] = num_distinct_prefix - num_distinct_suffix
        
        # Add nums[i] to current_suffix_elements. This prepares the set for the
        # next iteration (which will be for index i-1). For index i-1, the
        # suffix starts at index i, so nums[i] must be included in the count of distinct
        # elements for that suffix.
        current_suffix_elements.add(nums[i])
            
    return ans