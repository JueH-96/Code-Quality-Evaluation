class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = max(prefix[i], maxHeights[i])
            suffix[n - i] = max(suffix[n - i + 1], maxHeights[n - i])
        for i in range(n):
            suffix[i + 1] = max(suffix[i], suffix[i + 1])
        return max(prefix[i] + suffix[i + 1] for i in range(n))