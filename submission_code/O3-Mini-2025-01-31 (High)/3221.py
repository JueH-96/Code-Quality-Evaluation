from typing import List

class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        peaks = []
        # Traverse from the second element to the element before the last one.
        for i in range(1, len(mountain) - 1):
            if mountain[i] > mountain[i - 1] and mountain[i] > mountain[i + 1]:
                peaks.append(i)
        return peaks