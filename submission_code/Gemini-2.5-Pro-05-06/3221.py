from typing import List

class Solution:
  def findPeaks(self, mountain: List[int]) -> List[int]:
    peak_indices = []
    n = len(mountain)

    # A peak cannot be the first or the last element.
    # So, we iterate from the second element (index 1)
    # up to the second-to-last element (index n-2).
    # The constraint mountain.length >= 3 ensures n >= 3.
    # Thus, n-1 >= 2. The range for i, which is range(1, n-1), is valid.
    # For n=3, range(1,2) means i=1. This is the only possible internal index.
    for i in range(1, n - 1):
        # Check if the current element is strictly greater than its left neighbor
        # and strictly greater than its right neighbor.
        if mountain[i] > mountain[i-1] and mountain[i] > mountain[i+1]:
            peak_indices.append(i)
            
    return peak_indices