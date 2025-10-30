from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        ans = 0
        
        # Try each position i as the peak
        for i in range(n):
            # Start sum with the peak height
            total = maxHeights[i]
            
            # Build to the left of i, ensuring non-decreasing up to i
            last = maxHeights[i]
            for j in range(i - 1, -1, -1):
                last = min(last, maxHeights[j])
                total += last
            
            # Build to the right of i, ensuring non-increasing from i
            last = maxHeights[i]
            for j in range(i + 1, n):
                last = min(last, maxHeights[j])
                total += last
            
            # Track the maximum sum over all possible peaks
            ans = max(ans, total)
        
        return ans