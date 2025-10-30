from typing import List

class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        # Create an empty list to store the indices of the peaks.
        peaks = []

        # Iterate through the array elements that could potentially be peaks.
        # A peak cannot be the first (index 0) or the last element (index len(mountain)-1).
        # So, we check indices from 1 up to len(mountain) - 2.
        # In Python's range function, range(start, stop) includes 'start' but excludes 'stop'.
        # Thus, range(1, len(mountain) - 1) iterates through indices 1, 2, ..., len(mountain) - 2.
        # The minimum length of the array is 3, so len(mountain) - 1 will be at least 2.
        # range(1, 2) will yield i=1, which is correct for a length 3 array.
        for i in range(1, len(mountain) - 1):
            # Check if the current element (at index i) is strictly greater than its neighbors.
            # A peak must be strictly greater than the element to its left (index i-1)
            # AND strictly greater than the element to its right (index i+1).
            if mountain[i] > mountain[i-1] and mountain[i] > mountain[i+1]:
                # If both conditions are true, the element at index i is a peak.
                # Add the index i to our list of peaks.
                peaks.append(i)

        # After checking all potential peak locations, return the list of found peak indices.
        return peaks