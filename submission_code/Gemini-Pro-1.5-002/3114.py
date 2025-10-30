class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        ans = 0
        for i in range(n):
            heights = [0] * n
            heights[i] = maxHeights[i]
            for j in range(i - 1, -1, -1):
                heights[j] = min(maxHeights[j], heights[j + 1])
            for j in range(i + 1, n):
                heights[j] = min(maxHeights[j], heights[j - 1])
            ans = max(ans, sum(heights))
        return ans