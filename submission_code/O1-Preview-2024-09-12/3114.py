class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        for peak in range(n):
            heights = [0] * n
            heights[peak] = maxHeights[peak]
            # Left side
            for i in range(peak - 1, -1, -1):
                heights[i] = max(1, min(heights[i+1], maxHeights[i]))
            # Right side
            for i in range(peak +1, n):
                heights[i] = max(1, min(heights[i-1], maxHeights[i]))
            total = sum(heights)
            max_sum = max(max_sum, total)
        return max_sum