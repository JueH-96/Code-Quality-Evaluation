class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        best_sum = 0
        
        for peak in range(n):
            # We'll build a "mountain" with the peak at index `peak`
            
            # Start by setting the height at the peak to its maximum allowed
            heights = [0] * n
            heights[peak] = maxHeights[peak]
            
            # Fill to the left (non-decreasing when moving left->right)
            for i in range(peak - 1, -1, -1):
                heights[i] = min(heights[i + 1], maxHeights[i])
                # Ensure it's at least 1
                if heights[i] < 1:
                    heights[i] = 1
            
            # Fill to the right (non-increasing when moving left->right)
            for i in range(peak + 1, n):
                heights[i] = min(heights[i - 1], maxHeights[i])
                # Ensure it's at least 1
                if heights[i] < 1:
                    heights[i] = 1
            
            # Compute the sum of this mountain configuration
            current_sum = sum(heights)
            best_sum = max(best_sum, current_sum)
        
        return best_sum