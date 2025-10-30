from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        best = 0
        
        # Try each index as the mountain peak.
        for peak in range(n):
            # Build the best configuration for left part (indices 0 to peak)
            left = [0] * (peak + 1)
            left[peak] = maxHeights[peak]
            for i in range(peak - 1, -1, -1):
                # We want the left part to be as high as possible but non-decreasing
                left[i] = min(maxHeights[i], left[i + 1])
            
            # Build the best configuration for right part (indices peak to n-1)
            right = [0] * (n - peak)
            right[0] = maxHeights[peak]
            for i in range(1, n - peak):
                # To keep the non-increasing order on the right side,
                # each tower can be at most as high as the previous one
                right[i] = min(maxHeights[peak + i], right[i - 1])
            
            # The peak at 'peak' index is counted in both parts, so remove one copy.
            current_sum = sum(left) + sum(right[1:])
            best = max(best, current_sum)
        return best