import collections
from typing import List

class Solution:
  def longestEqualSubarray(self, nums: List[int], k: int) -> int:
    """
    Finds the length of the longest possible equal subarray after deleting at most k elements.

    The core idea is to rephrase the problem for a single target number, `val`.
    If we want to form an equal subarray of `val`s of length `L`, we need to pick `L`
    occurrences of `val` from the original `nums` array. To make them a contiguous
    subarray in the final array, the elements we pick must have been "brought together"
    by deleting the elements between them. The number of deleted elements is the
    total number of elements in the original span minus the `L` elements we keep.

    Let's say `val` appears at indices `i_1, i_2, ..., i_m`. If we choose to keep `L`
    occurrences starting from the `j`-th one, these are at original indices
    `indices[j], indices[j+1], ..., indices[j+L-1]`.
    This group of `val`s is contained within the original subarray that starts at
    `indices[j]` and ends at `indices[j+L-1]`.
    The length of this original segment is `indices[j+L-1] - indices[j] + 1`.
    The number of elements to delete is `(length of segment) - (number of val's kept)`.
    This is `(indices[j+L-1] - indices[j] + 1) - L`. This must be at most `k`.

    This subproblem can be solved for each unique number in `nums`. We can then take
    the maximum length found across all numbers.

    Algorithm:
    1. Group indices by number. We use a dictionary `pos` where `pos[num]` is a list
       of indices where `num` appears.
    2. For each number, we use a sliding window (two pointers, `left` and `right`)
       on its list of indices to find the longest valid sequence of occurrences.
    3. The window `indices[left...right]` represents a group of `val`s we might keep.
       - The number of `val`s is `current_len = right - left + 1`.
       - The number of elements to delete is `(indices[right] - indices[left] + 1) - current_len`.
       - We expand the window by moving `right`.
       - If the number of deletions exceeds `k`, we shrink the window by moving `left`.
       - We track the maximum `current_len` found across all valid windows.
    4. The overall time complexity is O(N) because each index is visited a constant
       number of times by the `left` and `right` pointers. The space complexity is O(N)
       for the `pos` map.
    """
    pos = collections.defaultdict(list)
    for i, num in enumerate(nums):
        pos[num].append(i)

    max_len = 0
    for indices in pos.values():
        n = len(indices)
        if n <= max_len:
            # Optimization: Not enough occurrences to beat the current max_len.
            continue

        left = 0
        for right in range(n):
            # Calculate the number of elements that are not our target number
            # within the span of the current window of occurrences.
            # Span length: indices[right] - indices[left] + 1
            # Number of target elements in span: right - left + 1
            # Number of other elements (to be deleted), simplified:
            deletions_needed = indices[right] - indices[left] - (right - left)

            # If the window requires too many deletions, shrink it from the left.
            while deletions_needed > k:
                left += 1
                deletions_needed = indices[right] - indices[left] - (right - left)

            # The current window is valid. Its length is a candidate for the answer.
            current_len = right - left + 1
            max_len = max(max_len, current_len)

    return max_len