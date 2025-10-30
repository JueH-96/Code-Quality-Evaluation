from typing import List
from collections import Counter

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        """
        Finds the length of the longest possible equal subarray after deleting
        at most k elements from nums.

        A subarray is considered "equal" if all its elements are the same.
        The problem can be reframed as finding the longest contiguous subarray
        nums[i...j] in the original array such that the number of elements
        within this subarray that are NOT equal to the most frequent element
        in this subarray is at most k.

        If nums[i...j] has length (j - i + 1) and the most frequent element
        appears max_freq times, the number of elements not equal to the most
        frequent is (j - i + 1) - max_freq. We need this to be <= k.
        If this condition holds, the max_freq occurrences of the most frequent
        element within nums[i...j] can form an equal subarray of length max_freq
        after deleting the other (j - i + 1) - max_freq elements within this range.
        The total deletions across the whole array is not explicitly limited to
        just within the [i, j] range, but the problem setup implies we choose a
        segment from the original array and delete elements to make it equal.
        The elements deleted are the ones within the segment [i, j] that we don't keep.
        Elements outside [i, j] are implicitly deleted as they are not part of
        the resulting equal subarray. The number of deletions we *count* towards
        the budget 'k' are those needed to make the chosen elements equal.
        The interpretation consistent with sliding window solutions to similar problems
        is that we select a contiguous window [i, j] in the original array,
        identify the most frequent element in this window, and consider deleting
        all other elements *within this window*. The cost is the count of these
        other elements. This count must be <= k. The length of the resulting
        equal subarray is the count of the most frequent element in the window.

        We use a sliding window [i, j]. We expand the window with j. We maintain
        the frequency counts of elements in the current window and the maximum
        frequency among them. If the window violates the condition
        (j - i + 1) - max_freq > k, we shrink the window from the left by moving i.
        The answer is the maximum max_freq found across all valid windows.

        Args:
            nums: The input list of integers.
            k: The maximum number of elements to delete.

        Returns:
            The length of the longest possible equal subarray.
        """
        n = len(nums)
        if n == 0:
            return 0

        # Sliding window [i, j]
        i = 0
        # counts store frequency of numbers in the current window [i, j]
        counts = Counter()
        # max_freq stores the maximum frequency of any number
        # in the current window [i, j]. This variable is updated
        # only when the window expands, reflecting the max frequency
        # in the window *at that point*. It is not necessarily the
        # true max frequency after shrinking.
        max_freq = 0
        
        # ans stores the maximum length of a valid equal subarray found so far
        # This corresponds to the maximum value of max_freq achieved
        # when the window [i, j] was valid.
        ans = 0

        # Iterate with the right pointer j
        for j in range(n):
            # Expand the window by including nums[j]
            counts[nums[j]] += 1
            
            # Update the maximum frequency in the current window [i, j]
            # This max_freq variable represents the maximum frequency
            # among all elements within the window [i, j] *after* nums[j]
            # was just added.
            max_freq = max(max_freq, counts[nums[j]])

            # The number of elements in the current window [i, j] that are NOT
            # the most frequent element is (window_length - max_freq).
            # Window length is (j - i + 1).
            # We need (j - i + 1) - max_freq <= k.
            # If (j - i + 1) - max_freq > k, the current window [i, j] is
            # invalid according to the constraint. We need to shrink it
            # from the left by moving the left pointer i.
            # We use the 'max_freq' value calculated *before* shrinking
            # in the condition check.
            while (j - i + 1) - max_freq > k:
                # Remove nums[i] from the window
                counts[nums[i]] -= 1
                # Move the left pointer
                i += 1
                # Note: We DO NOT recompute max_freq here. The condition
                # relies on the 'max_freq' from step 2 before entering
                # the while loop. By increasing i, the window size (j - i + 1)
                # decreases, making the condition easier to satisfy relative
                # to the (potentially outdated) max_freq. The loop terminates
                # when the window [i, j] is small enough such that the number
                # of non-most-frequent elements (if the max frequency were
                # still 'max_freq') is at most k.
                pass # max_freq is NOT recomputed here

            # After shrinking (if needed), the current window [i, j] is valid.
            # The length of the equal subarray we can form from this window
            # is the maximum frequency of an element within this window [i, j].
            # The 'max_freq' variable (calculated at step 2) holds the maximum
            # frequency in the window *before* potentially shrinking.
            # The key insight is that the maximum possible length of the equal
            # subarray we can get from any window ending at 'j' starting at
            # or after the initial 'i' is bounded by the 'max_freq' we tracked.
            # If the window [i, j] is valid (after shrinking), the maximum
            # frequency in this window is exactly `max(counts.values())`.
            # The variable `max_freq` (from step 2) is `max(counts.values())`
            # *before* shrinking. It turns out the maximum value `max_freq`
            # attains during the entire loop process is the answer.
            ans = max(ans, max_freq)

        return ans