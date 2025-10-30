class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        peaks = []
        # Iterate over the array from the second element to the second last element
        for i in range(1, len(mountain) - 1):
            # Check if the current element is greater than its neighbors
            if mountain[i] > mountain[i - 1] and mountain[i] > mountain[i + 1]:
                peaks.append(i)
        return peaks