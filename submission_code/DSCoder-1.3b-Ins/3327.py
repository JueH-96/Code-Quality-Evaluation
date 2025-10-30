class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        prefix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        dp = [[0]*(n+1) for _ in range(n+1)]
        for l in range(1, n+1):
            for i in range(n-l+1):
                j = i+l-1
                dp[i][j] = float('inf')
                for x in range(i, j+1):
                    y = prefix[j+1]-prefix[x+1]-prefix[i]+prefix[x]-prefix[i]+nums[x]
                    if x != j:
                        dp[i][j] = min(dp[i][j], dp[i][x]+dp[x+1][j])
                    dp[i][j] = min(dp[i][j], dp[i][x]+y+1)
        return dp[0][n]