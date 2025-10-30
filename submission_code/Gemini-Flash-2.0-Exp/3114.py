class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        for i in range(n):
            heights = [0] * n
            heights[i] = maxHeights[i]
            current_sum = heights[i]

            # Build left side
            for j in range(i - 1, -1, -1):
                heights[j] = min(maxHeights[j], heights[j + 1])
                current_sum += heights[j]

            # Build right side
            for j in range(i + 1, n):
                heights[j] = min(maxHeights[j], heights[j - 1])
                current_sum += heights[j]

            max_sum = max(max_sum, current_sum)

        return max_sum