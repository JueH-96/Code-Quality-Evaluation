import collections
from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # left pointer of the sliding window
        left = 0
        
        # frequency map for elements in the current window [left, right]
        counts = collections.defaultdict(int)
        
        # Stores the maximum frequency of any element in the current window [left, right].
        # This variable is crucial. It represents the count of the "target" identical element
        # if we were to try to make the current window an "equal subarray".
        max_freq_in_window = 0
        
        # Stores the maximum length of an equal subarray found so far across all valid windows.
        # This will be our final answer.
        max_length_achieved = 0
        
        # Iterate with the right pointer through the array
        for right in range(n):
            # 1. Expand the window: Add the current element `nums[right]` to the window.
            counts[nums[right]] += 1
            
            # 2. Update `max_freq_in_window`: After adding `nums[right]`, `max_freq_in_window`
            # must be updated to reflect the new maximum frequency of any element in the
            # current window `[left, right]`.
            max_freq_in_window = max(max_freq_in_window, counts[nums[right]])
            
            # 3. Check window validity and shrink if necessary:
            # The condition for a valid window `[left, right]` is that the number of elements
            # *not* equal to the most frequent element in this window must be at most `k`.
            # Number of elements to delete = (Current window length) - (Max frequency in window)
            # Current window length = `(right - left + 1)`
            # So, we need `(right - left + 1) - max_freq_in_window <= k`.
            
            # If the current window is invalid (i.e., requires more than `k` deletions),
            # shrink the window from the left by moving the `left` pointer.
            while (right - left + 1) - max_freq_in_window > k:
                # Remove `nums[left]` from the counts, as it's no longer in the window.
                counts[nums[left]] -= 1
                
                # Move the left pointer one step to the right.
                left += 1
                
                # IMPORTANT: `max_freq_in_window` is NOT updated here.
                # It continues to hold the peak frequency observed when the window expanded.
                # This works because we are maximizing `max_freq_in_window`. If the current window,
                # even after shrinking, leads to a smaller actual `max_freq_in_window`, it won't
                # improve our `max_length_achieved` anyway. The `while` condition correctly
                # identifies windows that are "too expensive" based on the current `max_freq_in_window` value,
                # ensuring that when we exit the loop, we have a potentially optimal `max_freq_in_window`
                # that satisfied the `k` constraint at some point.
            
            # 4. Update the overall maximum length achieved:
            # At this point, the window `[left, right]` is guaranteed to be valid according to
            # the `max_freq_in_window` value it carries. This `max_freq_in_window` is a candidate
            # for the longest equal subarray.
            max_length_achieved = max(max_length_achieved, max_freq_in_window)
            
        return max_length_achieved