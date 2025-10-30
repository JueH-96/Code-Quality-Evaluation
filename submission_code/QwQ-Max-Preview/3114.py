class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_total = 0
        
        for i in range(n):
            heights = [0] * n
            heights[i] = maxHeights[i]
            
            # Construct the left part (non-decreasing up to the peak)
            for j in range(i - 1, -1, -1):
                heights[j] = min(maxHeights[j], heights[j + 1])
            
            # Construct the right part (non-increasing from the peak)
            for j in range(i + 1, n):
                heights[j] = min(maxHeights[j], heights[j - 1])
            
            current_sum = sum(heights)
            if current_sum > max_total:
                max_total = current_sum
        
        return max_total