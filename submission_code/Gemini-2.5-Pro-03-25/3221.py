import sys
# Setting higher recursion depth for potential complex scenarios, although not strictly needed for this iterative solution.
sys.setrecursionlimit(10**6) 
from typing import List

class Solution:
    """
    This class provides a method to find all peak indices in a given mountain array.
    """
    def findPeaks(self, mountain: List[int]) -> List[int]:
        """
        Finds all the peaks in the mountain array.

        A peak is defined as an element that is strictly greater than its neighboring elements.
        The first and last elements of the array cannot be peaks.

        Args:
            mountain: A list of integers representing the mountain heights. 
                      Constraints: 3 <= len(mountain) <= 100, 1 <= mountain[i] <= 100.

        Returns:
            A list of integers containing the indices of the peaks in the mountain array.
            The order of indices in the returned list does not matter.
        """
        
        peaks_indices = [] # Initialize an empty list to store the indices of the peaks
        n = len(mountain) # Get the length of the input array
        
        # Iterate through the array elements, excluding the first and the last elements.
        # The loop runs from index 1 up to index n-2 (inclusive).
        # This is because the first (index 0) and last (index n-1) elements cannot be peaks.
        for i in range(1, n - 1):
            # Check if the current element mountain[i] is strictly greater than its left neighbor mountain[i-1]
            # and strictly greater than its right neighbor mountain[i+1].
            if mountain[i] > mountain[i-1] and mountain[i] > mountain[i+1]:
                # If both conditions are true, the element at index i is a peak.
                # Add the index i to the list of peak indices.
                peaks_indices.append(i)
                
        # Return the list containing all found peak indices.
        return peaks_indices