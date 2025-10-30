class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        max_subarrays = 1

        for i in range(1, n + 1):
            min_score = float('inf')
            for j in range(i, 0, -1):
                dp[j][i] = dp[j][i - 1] & nums[i - 1]
                min_score = min(min_score, dp[j][i])
                if min_score == 0:
                    max_subarrays = max(max_subarrays, i - j + 1)

        return max_subarrays