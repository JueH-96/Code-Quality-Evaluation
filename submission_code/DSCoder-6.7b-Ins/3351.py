class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        n = len(happiness)
        happiness.sort()
        dp = [[0] * (k + 1) for _ in range(2)]
        for i in range(n - 1, -1, -1):
            for j in range(k + 1):
                dp[i % 2][j] = max(
                    (dp[(i + 1) % 2][j] if j > 0 else 0),
                    (dp[(i + 1) % 2][j - 1] if j > 0 else 0) + happiness[i]
                )
        return dp[n % 2][k]