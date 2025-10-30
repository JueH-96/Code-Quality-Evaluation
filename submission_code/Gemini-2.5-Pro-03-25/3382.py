import collections
import bisect
from typing import List

class Solution:
  """
  This class provides a solution to count subarrays where the first and last elements
  are equal to the maximum element in the subarray.
  """
  def numberOfSubarrays(self, nums: List[int]) -> int:
      """
      Calculates the number of valid subarrays based on the problem criteria.
      A subarray nums[i:j+1] (indices from i to j inclusive) is valid if 
      nums[i] == nums[j] and nums[i] is the maximum value in the subarray nums[i...j].

      Args:
          nums: A list of positive integers.

      Returns:
          The count of subarrays satisfying the condition.
      """
      n = len(nums)
      # If the input array is empty, there are no subarrays.
      if n == 0:
          return 0
      
      # Step 1: Compute the 'previous greater element' index for each element in the array.
      # prev_greater[j] stores the index k < j such that nums[k] > nums[j].
      # If no such k exists (i.e., all elements to the left of j are <= nums[j]), prev_greater[j] is -1.
      # This is computed efficiently using a monotonic stack.
      prev_greater = [-1] * n
      # The stack stores indices. It maintains indices `k` such that `nums[k]` values are strictly decreasing.
      stack = [] 
      for j in range(n):
          # While the stack is not empty and the value at the index on top of the stack 
          # is less than or equal to the current value nums[j]:
          # Pop the index from the stack. The element at the popped index cannot be the 
          # 'previous greater' for nums[j] because nums[j] is greater or equal. 
          # It also cannot be the 'previous greater' for any element to the right of j 
          # that is greater than or equal to nums[j].
          while stack and nums[stack[-1]] <= nums[j]:
              stack.pop()
          
          # After popping, if the stack is not empty, the index at the top `k = stack[-1]`
          # satisfies `nums[k] > nums[j]`. This `k` is the nearest index to the left of `j` 
          # with a value strictly greater than nums[j]. Store this index.
          if stack:
              prev_greater[j] = stack[-1]
          # If the stack is empty, it means no element to the left of j is strictly greater than nums[j].
          # `prev_greater[j]` remains the default value -1.
          
          # Push the current index `j` onto the stack. This maintains the monotonic decreasing property 
          # (based on values) required for future calculations.
          stack.append(j)
          
      # Step 2: Store indices for each distinct value present in the array.
      # `indices_map` is a dictionary where keys are values from `nums` and values are sorted lists 
      # of indices where that key appears in `nums`.
      indices_map = collections.defaultdict(list)
      for idx, val in enumerate(nums):
          indices_map[val].append(idx)
          
      # Step 3: Calculate the total count of valid subarrays.
      total_count = 0
      
      # Iterate through each index `j` from 0 to n-1. Consider `j` as the right endpoint 
      # of potential valid subarrays.
      for j in range(n):
          # The value at index `j`. For a subarray `nums[i...j]` to be valid, `nums[j]` must be
          # equal to `nums[i]` and be the maximum value in the subarray.
          current_val = nums[j]
          
          # Get the index `pg = prev_greater[j]`. This is the index of the nearest element to the 
          # left of `j` which is strictly greater than `current_val`.
          # A valid start index `i` for the subarray must be greater than `pg`. 
          # This condition `i > pg` ensures that there is no element `nums[k]` with `k` in `[i, j)`
          # such that `nums[k] > current_val`. Since `nums[j] = current_val`, this implies `current_val`
          # is the maximum element in `nums[i...j]`.
          pg = prev_greater[j] 
          
          # Retrieve the sorted list of indices where `current_val` appears in `nums`.
          idx_list = indices_map[current_val]
          
          # We need to count how many indices `i` are present in `idx_list` such that `pg < i <= j`.
          # Each such index `i` corresponds to a valid start index for a subarray ending at `j`.
          
          # Use binary search for efficient counting. Python's `bisect.bisect_right` function is suitable.
          # `bisect_right(a, x)` finds an insertion point `k` such that all `a[p] <= x` for `p < k`.
          # Effectively, it gives `1 + index of the rightmost element <= x`, which is the count of elements <= x.
          
          # Find the count of indices `i` in `idx_list` such that `i <= j`.
          # This tells us how many times `current_val` has appeared up to index `j` (inclusive).
          count_le_j = bisect.bisect_right(idx_list, j)
          
          # Find the count of indices `i` in `idx_list` such that `i <= pg`.
          # This tells us how many times `current_val` has appeared up to index `pg` (inclusive).
          count_le_pg = bisect.bisect_right(idx_list, pg)
          
          # The difference `count_le_j - count_le_pg` gives the exact number of indices `i` in `idx_list`
          # that fall within the range `(pg, j]`. These are the valid start indices `i` for subarrays ending at `j`.
          count_for_j = count_le_j - count_le_pg
          
          # Add the count found for the current endpoint `j` to the overall total count.
          total_count += count_for_j
          
      # Return the final computed total count of valid subarrays.
      return total_count