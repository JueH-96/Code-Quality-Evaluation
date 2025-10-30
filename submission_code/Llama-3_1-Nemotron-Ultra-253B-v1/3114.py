from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        for i in range(n):
            # Calculate left part (non-decreasing up to i)
            left_sum = 0
            prev = maxHeights[i]
            left_sum += prev
            for j in range(i - 1, -1, -1):
                current = min(maxHeights[j], prev)
                left_sum += current
                prev = current
            
            # Calculate right part (non-increasing from i)
            right_sum = 0
            prev = maxHeights[i]
            right_sum += prev
            for j in range(i + 1, n):
                current = min(maxHeights[j], prev)
                right_sum += current
                prev = current
            
            # Total sum for this peak, subtract the peak counted twice
            total = left_sum + right_sum - maxHeights[i]
            if total > max_sum:
                max_sum = total
        
        return max_sum