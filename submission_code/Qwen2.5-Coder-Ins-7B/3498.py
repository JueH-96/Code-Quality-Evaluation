class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        groups = [[] for _ in range(k)]
        for i in range(n):
            groups[nums[i] % k].append(i)
        
        dp = [[float('inf')] * k for _ in range(n // 2 + 1)]
        dp[0][0] = 0
        
        for i in range(1, n // 2 + 1):
            for j in range(k):
                dp[i][j] = min(dp[i][j], dp[i - 1][j])
                for g in groups[j]:
                    if i - 1 - g // 2 >= 0:
                        dp[i][j] = min(dp[i][j], dp[i - 1 - g // 2][g % k] + g // 2)
        
        return dp[n // 2][0]