from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        best = 0
        
        # Try each index p as the peak
        for p in range(n):
            peak = maxHeights[p]
            total = peak
            
            # Build left side (non-decreasing into the peak,
            # but when we walk from p to 0 it's non-increasing)
            cur = peak
            for i in range(p - 1, -1, -1):
                cur = min(cur, maxHeights[i])
                total += cur
            
            # Build right side (non-increasing away from the peak)
            cur = peak
            for i in range(p + 1, n):
                cur = min(cur, maxHeights[i])
                total += cur
            
            best = max(best, total)
        
        return best