class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        peaks = []
        # Start from index 1 and go until the second last element
        for i in range(1, len(mountain) - 1):
            if mountain[i] > mountain[i - 1] and mountain[i] > mountain[i + 1]:
                peaks.append(i)
        return peaks