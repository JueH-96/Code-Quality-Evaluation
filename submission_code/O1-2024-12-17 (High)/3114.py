class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        ans = 0
        
        for peak_index in range(n):
            # Set the peak to the maximum allowed at this index
            peak_value = maxHeights[peak_index]
            
            # Build left side (non-decreasing up to peak_index)
            left_heights = [0] * n
            left_heights[peak_index] = peak_value
            for j in range(peak_index - 1, -1, -1):
                left_heights[j] = min(maxHeights[j], left_heights[j + 1])
            
            # Build right side (non-increasing from peak_index)
            right_heights = [0] * n
            right_heights[peak_index] = peak_value
            for j in range(peak_index + 1, n):
                right_heights[j] = min(maxHeights[j], right_heights[j - 1])
            
            # Combine: use left side for indices < peak_index, 
            # and right side from peak_index onward.
            current_sum = sum(left_heights[:peak_index]) + sum(right_heights[peak_index:])
            ans = max(ans, current_sum)
        
        return ans