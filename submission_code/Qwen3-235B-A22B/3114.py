from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        for i in range(n):
            # Compute left array for peak at i
            left = []
            left.append(maxHeights[0])
            for j in range(1, i + 1):
                current = maxHeights[j]
                prev = left[-1]
                if current >= prev:
                    left.append(current)
                else:
                    left.append(current)
                    k = len(left) - 2  # Index of the previous element after appending
                    while k >= 0 and left[k] > current:
                        left[k] = current
                        k -= 1
            
            # Compute right array starting at i
            peak_val = left[-1]
            right = [peak_val]
            for j in range(i + 1, n):
                prev = right[-1]
                current_val = min(maxHeights[j], prev)
                right.append(current_val)
            
            # Calculate the total sum for this peak
            sum_total = sum(left) + sum(right) - peak_val
            if sum_total > max_sum:
                max_sum = sum_total
        
        return max_sum