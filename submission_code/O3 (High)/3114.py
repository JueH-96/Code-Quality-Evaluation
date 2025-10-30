from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        """
        For every position we treat it as the peak of the mountain.
        Starting from the peak we extend to the left and to the right,
        each time the height of the next tower can be at most the
        minimum of its own limit (maxHeights[i]) and the height we
        have just placed next to it.  That greedy choice is optimal
        because increasing any already-fixed height can never hurt the
        feasibility and only increases the total sum.

        n ≤ 1000, so an O(n²) enumeration of all peaks is fast enough.
        """
        n = len(maxHeights)
        best = 0

        for peak in range(n):
            total = maxHeights[peak]

            # build to the left
            prev = maxHeights[peak]
            for i in range(peak - 1, -1, -1):
                prev = min(prev, maxHeights[i])
                total += prev

            # build to the right
            prev = maxHeights[peak]
            for i in range(peak + 1, n):
                prev = min(prev, maxHeights[i])
                total += prev

            best = max(best, total)

        return best