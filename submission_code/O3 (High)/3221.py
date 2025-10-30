from typing import List

class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        """
        Return a list of indices i such that:
          - 0 < i < len(mountain)-1   (i is not the first or last index)
          - mountain[i] > mountain[i-1] and mountain[i] > mountain[i+1]
        The returned list may be in any order; here we collect them in increasing order.
        """
        n = len(mountain)
        peaks = []
        
        # Iterate over all valid candidate positions (exclude first and last)
        for i in range(1, n - 1):
            if mountain[i] > mountain[i - 1] and mountain[i] > mountain[i + 1]:
                peaks.append(i)
                
        return peaks