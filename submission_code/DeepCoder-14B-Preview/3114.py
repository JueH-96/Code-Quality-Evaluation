class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_total = 0
        
        for i in range(n):
            # Compute the left part
            left = [0] * n
            left[i] = maxHeights[i]
            for j in range(i - 1, -1, -1):
                left[j] = min(maxHeights[j], left[j + 1])
            
            # Compute the right part
            right = [0] * n
            right[i] = maxHeights[i]
            for j in range(i + 1, n):
                right[j] = min(maxHeights[j], right[j - 1])
            
            # Calculate the total sum for this peak
            sum_left = sum(left[:i + 1])
            sum_right = sum(right[i:])
            total = sum_left + sum_right - left[i]  # Subtract the peak once
            
            if total > max_total:
                max_total = total
        
        return max_total