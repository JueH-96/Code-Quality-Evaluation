class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        coords = sorted(coordinates)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if coords[j][0] < coords[i][0] and coords[j][1] < coords[i][1]:
                    dp[i] = max(dp[i], dp[j] + 1)

        ans = 0
        for i in range(n):
            if coords[i] == coordinates[k]:
                ans = dp[i]
                break
        return ans