class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        peaks = []
        
        # Check elements from index 1 to n-2 (excluding first and last)
        for i in range(1, len(mountain) - 1):
            # Check if current element is strictly greater than both neighbors
            if mountain[i] > mountain[i-1] and mountain[i] > mountain[i+1]:
                peaks.append(i)
        
        return peaks