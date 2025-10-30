class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        for peak in range(n):
            total_sum = maxHeights[peak]
            
            # Build left part (non-decreasing towards peak)
            current_height = maxHeights[peak]
            for i in range(peak - 1, -1, -1):
                current_height = min(current_height, maxHeights[i])
                total_sum += current_height
            
            # Build right part (non-increasing from peak)
            current_height = maxHeights[peak]
            for i in range(peak + 1, n):
                current_height = min(current_height, maxHeights[i])
                total_sum += current_height
            
            max_sum = max(max_sum, total_sum)
        
        return max_sum