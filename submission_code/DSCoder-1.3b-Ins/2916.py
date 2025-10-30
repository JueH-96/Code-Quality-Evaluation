class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        dp = [False] * n
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
            if prefix_sum[i + 1] >= m:
                dp[i] = True
        for i in range(n - 1, -1, -1):
            if dp[i]:
                if i + 2 < n and prefix_sum[i + 2] < m:
                    continue
                for j in range(i + 1, -1, -1):
                    if prefix_sum[j] >= m:
                        dp[j] = True
                        break
        return dp[0]