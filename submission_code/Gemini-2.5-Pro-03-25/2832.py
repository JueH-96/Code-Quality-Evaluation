import collections
from typing import List

class Solution:
  """
  Finds the length of the longest possible equal subarray after deleting at most k elements.
  An equal subarray has all elements equal.
  """
  def longestEqualSubarray(self, nums: List[int], k: int) -> int:
    """
    Args:
      nums: A 0-indexed integer array.
      k: The maximum number of elements that can be deleted.

    Returns:
      The length of the longest possible equal subarray after deletions.
    """
    
    # Store indices for each number encountered in nums.
    # Using collections.defaultdict(list) provides a convenient way to group indices by number.
    # The keys will be the distinct numbers in nums, and the values will be lists containing the indices
    # where each number appears in nums. The indices within each list will be naturally sorted
    # because we iterate through nums in order.
    indices_map = collections.defaultdict(list)
    for i, num in enumerate(nums):
        indices_map[num].append(i)
        
    # Initialize the maximum length found so far to 0.
    # The problem asks for the length of the longest possible equal subarray.
    # If nums is non-empty, we can always form an equal subarray of length at least 1
    # (by potentially deleting elements around a single occurrence).
    # If k allows deleting all elements, we could form an empty subarray of length 0.
    # However, since nums is non-empty (constraint: 1 <= nums.length), we can always
    # find an equal subarray of length at least 1 unless k is so large that we choose to delete everything.
    # The maximum length logic correctly captures the largest possible size, which will be >= 0.
    # Based on examples, it seems we want the maximum length, even if 0 is possible.
    # Our approach will find the max length >= 1 if nums is non-empty. The base case 0 takes care of possibilities.
    max_length = 0
    
    # Iterate through each distinct number present in nums.
    # For each number, find the longest possible equal subarray consisting only of this number.
    for num in indices_map:
        indices = indices_map[num] # Get the list of indices where 'num' appears.
        
        # Use a sliding window approach on the list of indices for the current number 'num'.
        # 'left' is the start pointer of the window (index into the 'indices' list).
        # 'right' is the end pointer of the window (index into the 'indices' list).
        left = 0
        
        # Expand the window by moving the 'right' pointer one step at a time through the indices list.
        for right in range(len(indices)):
            # The current window represents a potential sequence of 'num' occurrences.
            # These occurrences are at original indices: indices[left], indices[left+1], ..., indices[right].
            
            # To form an equal subarray using these occurrences, we must consider the subarray
            # in the original 'nums' that spans from index indices[left] to indices[right].
            # The length of this span is `indices[right] - indices[left] + 1`.
            span_length = indices[right] - indices[left] + 1
            
            # The number of 'num' elements within this window (and span) is `right - left + 1`.
            count_of_num = right - left + 1
            
            # The number of elements within the span that are NOT 'num' must be deleted.
            # This count is `span_length - count_of_num`.
            deletions_needed = span_length - count_of_num
            
            # If the number of deletions needed exceeds the allowed budget 'k',
            # we must shrink the window from the left side by incrementing 'left'.
            # This process continues until the number of deletions needed is within the budget 'k'.
            # The `while` loop ensures the window `[left, right]` remains valid w.r.t `k`.
            while deletions_needed > k:
                # Move the left pointer one step to the right, effectively removing the element at indices[left]
                # from consideration for the current window.
                left += 1
                
                # Recalculate the properties of the shrunk window based on the new 'left' pointer.
                # We need to update span_length, count_of_num, and deletions_needed.
                span_length = indices[right] - indices[left] + 1
                count_of_num = right - left + 1
                deletions_needed = span_length - count_of_num
                # Alternative simpler update inside the while loop:
                # We just need to check the condition again. Recalculating `deletions_needed` is sufficient.
                # deletions_needed = (indices[right] - indices[left] + 1) - (right - left + 1)

            # After the while loop, the current window [left, right] corresponds to a valid scenario
            # where we can form an equal subarray of 'num' with length `count_of_num` by deleting
            # `deletions_needed` elements (where deletions_needed <= k).
            
            # Update the overall maximum length found across all numbers and all valid windows checked so far.
            # The length of the equal subarray formed is `count_of_num`.
            max_length = max(max_length, count_of_num)
            
    # Return the overall maximum length found.
    return max_length