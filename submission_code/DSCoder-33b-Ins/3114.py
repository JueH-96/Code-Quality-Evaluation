class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        heights = [0] * n
        heights[0] = min(maxHeights[0], 1)
        for i in range(1, n):
            heights[i] = min(maxHeights[i], heights[i - 1] + 1)
        for i in range(n - 2, -1, -1):
            heights[i] = min(maxHeights[i], heights[i + 1] + 1, heights[i])
        return sum(heights)