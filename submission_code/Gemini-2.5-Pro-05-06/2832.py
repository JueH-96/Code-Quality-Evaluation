import collections
from typing import List

class Solution:
  def longestEqualSubarray(self, nums: List[int], k: int) -> int:
    val_to_indices = collections.defaultdict(list)
    # Store indices for each number
    # e.g., if nums = [1,3,2,3,1,3], val_to_indices will be:
    # {1: [0, 4], 3: [1, 3, 5], 2: [2]}
    for i, num_val in enumerate(nums):
        val_to_indices[num_val].append(i)
    
    max_overall_length = 0
    
    # Constraints: nums.length >= 1.
    # This means there's always at least one element.
    # We can always form an equal subarray of length 1 (by picking any single element)
    # with 0 deletions. So, if nums is not empty, the answer is at least 1.
    # Initializing max_overall_length to 0 is fine; it will be updated to at least 1
    # for any non-empty nums (as shown by testing with nums=[X]).

    # Iterate over each unique number present in nums
    for val_num in val_to_indices:
        pos = val_to_indices[val_num] # List of indices where val_num appears
        
        # Optimization: If the total count of this number `val_num` (which is len(pos))
        # is not more than `max_overall_length` found so far, then `val_num`
        # cannot form a longer equal subarray. So, we can skip processing for this `val_num`.
        if len(pos) <= max_overall_length:
            continue
        
        left = 0 # Left pointer for the sliding window over the `pos` list
        # `right` pointer iterates from the beginning to the end of `pos` list
        for right in range(len(pos)):
            # `current_val_count` is the number of `val_num` elements we are trying to keep.
            # These are occurrences of `val_num` at original indices:
            # `pos[left], pos[left+1], ..., pos[right]`.
            # The count of such elements is `right - left + 1`.
            current_val_count = right - left + 1
            
            # `elements_to_delete` are those elements in the original `nums` array
            # that are located *between* `nums[pos[left]]` and `nums[pos[right]]` (inclusive segment)
            # but are *not* one of the `val_num`s we decided to keep (from `pos[left]` to `pos[right]`).
            # The length of this segment in the original array is `pos[right] - pos[left] + 1`.
            # From this segment, we are keeping `current_val_count` elements (all are `val_num`).
            # So, the number of other elements that must be deleted is:
            # (length of original segment) - (count of `val_num`s we keep in it).
            elements_to_delete = (pos[right] - pos[left] + 1) - current_val_count
            
            # If `elements_to_delete` exceeds `k`, our budget for deletions,
            # we must shrink the window from the left by incrementing `left`.
            # This continues until `elements_to_delete` is within budget (`<= k`).
            while elements_to_delete > k:
                left += 1
                # Recompute counts for the new, smaller window
                current_val_count = right - left + 1
                # The number of elements to delete must also be recomputed.
                # Note: As `left` increases, `pos[left]` increases (or `left` catches up to `right`).
                # `pos[right] - pos[left]` decreases. `current_val_count` decreases.
                # The overall `elements_to_delete` value tends to decrease.
                # The loop terminates because eventually `left` can become `right`.
                # In that case, `current_val_count` is 1, and `elements_to_delete` is 0.
                # Since `k >= 0` (a constraint), `0 > k` is false, so the loop stops.
                # Thus, `left` will not exceed `right`.
                elements_to_delete = (pos[right] - pos[left] + 1) - current_val_count
            
            # At this point, the window `pos[left...right]` defines a valid way to form
            # an equal subarray of `val_num`s. Its length is `current_val_count`.
            # We check if this is longer than any we've found so far.
            max_overall_length = max(max_overall_length, current_val_count)
            
    return max_overall_length