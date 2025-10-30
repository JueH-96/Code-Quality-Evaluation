from typing import List

class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        """
        This function finds all the peaks in the given mountain array.
        
        A peak is defined as an element that is strictly greater than its neighboring elements.
        The first and last elements of the array are not considered as peaks.
        
        Args:
        mountain (List[int]): A 0-indexed array of integers.
        
        Returns:
        List[int]: A list of indices of peaks in the given array.
        """
        
        # Initialize an empty list to store the indices of peaks
        peaks = []
        
        # Iterate over the array, excluding the first and last elements
        for i in range(1, len(mountain) - 1):
            # Check if the current element is strictly greater than its neighboring elements
            if mountain[i] > mountain[i - 1] and mountain[i] > mountain[i + 1]:
                # If it is, add its index to the list of peaks
                peaks.append(i)
        
        # Return the list of indices of peaks
        return peaks