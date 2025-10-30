from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        for i in range(n):
            # Compute sum for the left part (non-decreasing up to index i)
            sum_left = 0
            curr_val = maxHeights[i]
            for j in range(i, -1, -1):
                height = min(maxHeights[j], curr_val)
                sum_left += height
                curr_val = height  # Update the current value for the chain
            
            # Compute sum for the right part (non-increasing from index i)
            sum_right = 0
            curr_val_right = maxHeights[i]
            for k in range(i, n):
                height = min(maxHeights[k], curr_val_right)
                sum_right += height
                curr_val_right = height  # Update the current value for the chain
            
            # Total sum for this peak, correcting for double-counting height at index i
            current_sum = sum_left + sum_right - maxHeights[i]
            
            # Update the maximum sum found
            if current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum