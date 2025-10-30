import collections
from typing import List

class Solution:
    """
    Solves the problem of finding the minimum index for a valid split
    in an array with a dominant element.

    An element x of an integer array arr of length m is dominant if 
    freq(x) * 2 > m, where freq(x) is the number of occurrences of x in arr.
    Note that this definition implies that arr can have at most one dominant element.

    Given a 0-indexed integer array nums of length n with one dominant element.
    A split at index i is valid if:
    1. 0 <= i < n - 1
    2. The subarray nums[0, ..., i] has a dominant element `d_left`.
    3. The subarray nums[i + 1, ..., n - 1] has a dominant element `d_right`.
    4. `d_left` == `d_right`.

    It can be proven that if such a valid split exists, the common dominant element 
    `d_left` (`d_right`) must be the dominant element of the original array `nums`.

    The goal is to return the minimum index `i` of a valid split. If no valid split exists, return -1.
    """
    def minimumIndex(self, nums: List[int]) -> int:
        """
        Finds the minimum index i such that splitting nums at i results in two
        subarrays where the dominant element of nums is also dominant in both subarrays.

        Args:
            nums: A list of integers with exactly one dominant element.

        Returns:
            The minimum valid split index, or -1 if no such index exists.
        """
        n = len(nums)
        
        # According to constraints, n >= 1.
        # A valid split requires index i such that 0 <= i < n - 1. This is only possible if n >= 2.
        # If n = 1, the loop `range(n - 1)` below becomes `range(0)`, which is empty.
        # The function will correctly return -1 at the end for n=1.

        # Step 1: Find the dominant element and its total frequency in the original array `nums`.
        # `collections.Counter` is used for efficient frequency counting.
        counts = collections.Counter(nums)
        
        dominant_val = -1  # Variable to store the dominant value
        total_freq = 0     # Variable to store the frequency of the dominant value
        
        # Iterate through the element counts to identify the dominant element.
        # The problem statement guarantees that `nums` has exactly one dominant element.
        for val, freq in counts.items():
            # The dominance condition is freq(val) * 2 > n
            if freq * 2 > n:
                dominant_val = val
                total_freq = freq
                # Since there is exactly one dominant element, we can stop after finding it.
                break 
        
        # Step 2: Iterate through all possible valid split indices `i`.
        # A valid split index `i` must be in the range [0, n-2].
        current_freq_left = 0 # Tracks the frequency of `dominant_val` in the left subarray `nums[0...i]`
        
        # The loop iterates through indices from 0 up to n-2.
        for i in range(n - 1): 
            # Update the frequency count for the left subarray ending at index `i`.
            # If the current element `nums[i]` is the dominant element, increment the left count.
            if nums[i] == dominant_val:
                current_freq_left += 1

            # Calculate the lengths of the left and right subarrays for the split at index `i`.
            m_left = i + 1       # Length of the left subarray `nums[0...i]`
            m_right = n - m_left # Length of the right subarray `nums[i+1...n-1]`

            # Check Condition 1: Is `dominant_val` dominant in the left subarray?
            # The condition is: frequency_in_left * 2 > length_of_left
            is_dominant_in_left = current_freq_left * 2 > m_left

            # Check Condition 2: Is `dominant_val` dominant in the right subarray?
            # First, calculate the frequency of `dominant_val` in the right subarray.
            # frequency_in_right = total_frequency - frequency_in_left
            freq_right = total_freq - current_freq_left
            # The condition is: frequency_in_right * 2 > length_of_right
            is_dominant_in_right = freq_right * 2 > m_right

            # A split at index `i` is valid if `dominant_val` is dominant in both resulting subarrays.
            if is_dominant_in_left and is_dominant_in_right:
                # Since we are iterating `i` from 0 upwards, the first index `i` that satisfies 
                # both conditions is the minimum valid split index.
                return i

        # Step 3: If the loop completes without finding any valid split index,
        # it means no such split exists according to the given criteria. Return -1.
        return -1