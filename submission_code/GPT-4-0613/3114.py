class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        maxHeights.append(0)
        maxHeights.insert(0, 0)
        for i in range(n, 0, -1):
            maxHeights[i] = min(maxHeights[i], maxHeights[i + 1])
        for i in range(1, n + 1):
            maxHeights[i] = min(maxHeights[i], maxHeights[i - 1] + 1)
        return sum(maxHeights[1:n + 1])