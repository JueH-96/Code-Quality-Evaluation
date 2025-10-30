from typing import List

class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        """
        Finds all the peaks in a mountain array.

        A peak is an element that is strictly greater than its neighboring elements.
        The first and last elements of the array are not considered peaks.

        Args:
            mountain: A list of integers.

        Returns:
            A list of indices of the peaks.
        """
        
        # Initialize an empty list to store the indices of the peaks.
        peaks = []
        
        # Get the length of the array.
        n = len(mountain)
        
        # Iterate from the second element (index 1) to the second-to-last element (index n-2).
        # The first and last elements cannot be peaks, so they are skipped.
        for i in range(1, n - 1):
            # A peak is strictly greater than its left and right neighbors.
            if mountain[i] > mountain[i - 1] and mountain[i] > mountain[i + 1]:
                # If the condition is met, add the index to our list.
                peaks.append(i)
                
        return peaks