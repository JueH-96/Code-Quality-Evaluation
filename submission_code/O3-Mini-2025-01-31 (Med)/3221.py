from typing import List

class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        peaks = []
        # iterate from 1 to len(mountain)-2, as first and last cannot be peaks
        for i in range(1, len(mountain) - 1):
            if mountain[i] > mountain[i - 1] and mountain[i] > mountain[i + 1]:
                peaks.append(i)
        return peaks

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Test case 1
    mountain1 = [2, 4, 4]
    print(sol.findPeaks(mountain1))  # Expected output: []
    
    # Test case 2
    mountain2 = [1, 4, 3, 8, 5]
    print(sol.findPeaks(mountain2))  # Expected output: [1, 3]