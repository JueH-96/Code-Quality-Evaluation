class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0

        for peak_index in range(n):
            heights = [0] * n
            heights[peak_index] = maxHeights[peak_index]
            current_sum = heights[peak_index]

            # Construct left side
            for i in range(peak_index - 1, -1, -1):
                heights[i] = min(maxHeights[i], heights[i + 1])
                current_sum += heights[i]

            # Construct right side
            for i in range(peak_index + 1, n):
                heights[i] = min(maxHeights[i], heights[i - 1])
                current_sum += heights[i]

            max_sum = max(max_sum, current_sum)

        return max_sum