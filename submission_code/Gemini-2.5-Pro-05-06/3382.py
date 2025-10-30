from typing import List
import collections
import bisect

class Solution:
  def numberOfSubarrays(self, nums: List[int]) -> int:
    N = len(nums)
    # According to constraints N >= 1, so N=0 case is not strictly needed
    # but good for robustness if constraints were different.
    if N == 0:
        return 0

    # Calculate Next Greater Element (NGE) for each index
    # NGE[i] is the index k > i such that nums[k] > nums[i], and k is minimized.
    # If no such k exists, NGE[i] = N (an out-of-bounds index).
    nge = [N] * N
    stack = []  # Stores indices
    # Iterate from left to right. When considering nums[i]:
    # If nums[i] is greater than nums[stack.top()], then nums[i] is the NGE for stack.top().
    # Keep popping from stack while this condition holds.
    # Then push i onto stack. Stack elements s_1, s_2, ... from bottom to top
    # will have nums[s_1] >= nums[s_2] >= ... (monotonic non-increasing values).
    for i in range(N):
        while stack and nums[stack[-1]] < nums[i]: # Strictly smaller for NGE
            nge[stack.pop()] = i
        stack.append(i)
    
    # Store indices for each value. e.g., value_to_indices[5] = [0, 3, 10]
    value_to_indices = collections.defaultdict(list)
    for i, num in enumerate(nums):
        value_to_indices[num].append(i)

    # Initialize count. All single-element subarrays [nums[k]] are valid.
    # nums[k] == nums[k], and nums[k] is the largest in [nums[k]].
    # There are N such subarrays.
    count = N
    
    # Now, consider subarrays nums[i...j] where j > i.
    # For each element nums[i], consider it as the left endpoint of a subarray.
    for i in range(N):
        val = nums[i]  # The value that must be first, last, and maximum.
        
        # The subarray is nums[i...j]. We need nums[i] = nums[j] = val.
        # All elements nums[k] for i < k < j must satisfy nums[k] <= val.
        # The NGE[i] gives an index `limit_idx_nge` such that nums[limit_idx_nge] > val,
        # and for all p where i < p < limit_idx_nge, nums[p] <= val.
        # Thus, the right endpoint j must be < limit_idx_nge.
        limit_idx_nge = nge[i]
        
        # Get the list of all indices where `val` appears. This list is sorted.
        indices_for_val = value_to_indices[val]
        
        # We need to find indices `j` in `indices_for_val` such that:
        # 1. `j > i` (j must be to the right of i)
        # 2. `j < limit_idx_nge` (to ensure `val` remains the maximum)
        
        # `bisect_right(list, x)` finds an insertion point for `x` in `list` to maintain sorted order.
        # The insertion point `p` is such that all `e` in `list[:p]` have `e <= x`,
        # and all `e` in `list[p:]` have `e > x`.
        # So, `start_idx_in_list` will be the index in `indices_for_val` of the first element strictly greater than `i`.
        start_idx_in_list = bisect.bisect_right(indices_for_val, i)
        
        # `bisect_left(list, x)` finds an insertion point for `x`.
        # The insertion point `p` is such that all `e` in `list[:p]` have `e < x`,
        # and all `e` in `list[p:]` have `e >= x`.
        # So, `end_idx_in_list` will be the index in `indices_for_val` of the first element that is `>= limit_idx_nge`.
        # The elements we are interested in are `indices_for_val[k]` where `k` is in `[start_idx_in_list, end_idx_in_list - 1]`.
        end_idx_in_list = bisect.bisect_left(indices_for_val, limit_idx_nge)
        
        # The count of valid `j`s is the number of elements in the slice 
        # `indices_for_val[start_idx_in_list : end_idx_in_list]`.
        num_valid_j = end_idx_in_list - start_idx_in_list
        
        # Add this to the total count.
        count += num_valid_j
            
    return count