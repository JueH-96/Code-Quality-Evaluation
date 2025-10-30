from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        for i in range(n):
            # Calculate left sum (from 0 to i-1)
            left_sum = 0
            current = maxHeights[i]
            for j in range(i-1, -1, -1):
                current = min(current, maxHeights[j])
                left_sum += current
            
            # Calculate right sum (from i+1 to n-1)
            right_sum = 0
            current = maxHeights[i]
            for j in range(i+1, n):
                current = min(current, maxHeights[j])
                right_sum += current
            
            total = left_sum + maxHeights[i] + right_sum
            if total > max_sum:
                max_sum = total
        
        return max_sum