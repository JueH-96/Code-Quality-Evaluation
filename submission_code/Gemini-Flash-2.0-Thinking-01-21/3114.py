class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        for peak_index in range(n):
            heights = [0] * n
            heights[peak_index] = maxHeights[peak_index]
            for j in range(peak_index - 1, -1, -1):
                heights[j] = min(maxHeights[j], heights[j + 1])
            for j in range(peak_index + 1, n):
                heights[j] = min(maxHeights[j], heights[j - 1])
            current_sum = sum(heights)
            max_sum = max(max_sum, current_sum)
        return max_sum