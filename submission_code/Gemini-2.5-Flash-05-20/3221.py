from typing import List

class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        """
        Finds all peaks in the given mountain array.

        A peak is defined as an element that is strictly greater than its
        neighboring elements. The first and last elements of the array
        cannot be peaks.

        Args:
            mountain: A 0-indexed list of integers representing the mountain array.

        Returns:
            A list of indices of peaks in the given array. The order of indices
            does not matter.
        """
        peaks_indices = []
        n = len(mountain)

        # According to the problem definition, the first and last elements
        # cannot be peaks. Therefore, we only need to iterate from the
        # second element (index 1) up to the second-to-last element (index n-2).
        # This range ensures that for any element mountain[i],
        # mountain[i-1] (left neighbor) and mountain[i+1] (right neighbor)
        # always exist.
        for i in range(1, n - 1):
            # Check if the current element is strictly greater than its left neighbor
            is_greater_than_left = mountain[i] > mountain[i-1]
            
            # Check if the current element is strictly greater than its right neighbor
            is_greater_than_right = mountain[i] > mountain[i+1]

            # If both conditions are true, then mountain[i] is a peak
            if is_greater_than_left and is_greater_than_right:
                peaks_indices.append(i)
        
        return peaks_indices