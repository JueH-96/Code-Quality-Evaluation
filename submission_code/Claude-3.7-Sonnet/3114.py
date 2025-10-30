class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        for peak in range(n):
            current_sum = maxHeights[peak]
            
            # Calculate sum to the left of the peak
            prev_height = maxHeights[peak]
            for i in range(peak - 1, -1, -1):
                height = min(maxHeights[i], prev_height)
                current_sum += height
                prev_height = height
            
            # Calculate sum to the right of the peak
            prev_height = maxHeights[peak]
            for i in range(peak + 1, n):
                height = min(maxHeights[i], prev_height)
                current_sum += height
                prev_height = height
            
            max_sum = max(max_sum, current_sum)
        
        return max_sum