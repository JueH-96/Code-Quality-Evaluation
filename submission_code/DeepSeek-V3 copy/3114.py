class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        
        for peak in range(n):
            heights = [0] * n
            heights[peak] = min(maxHeights[peak], float('inf'))
            
            # Left of the peak
            for i in range(peak - 1, -1, -1):
                heights[i] = min(maxHeights[i], heights[i + 1])
            
            # Right of the peak
            for i in range(peak + 1, n):
                heights[i] = min(maxHeights[i], heights[i - 1])
            
            current_sum = sum(heights)
            if current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum