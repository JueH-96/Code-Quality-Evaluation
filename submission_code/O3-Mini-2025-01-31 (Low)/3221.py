from typing import List

class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        peaks = []
        for i in range(1, len(mountain) - 1):
            if mountain[i] > mountain[i - 1] and mountain[i] > mountain[i + 1]:
                peaks.append(i)
        return peaks

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test cases
    print(sol.findPeaks([2, 4, 4]))  # Output: []
    print(sol.findPeaks([1, 4, 3, 8, 5]))  # Output: [1, 3]