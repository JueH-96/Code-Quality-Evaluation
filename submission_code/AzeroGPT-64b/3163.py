class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        dp = [[0] * len(nums) for _ in range(len(nums))]
        ans = 0
        
        for i, v in enumerate(nums):
            dp[i][i] = 1
            ans += 1
            for j in range(i):
                if v not in nums[j:i]:
                    dp[j][i] = dp[j][i - 1] + 1
                    ans += (dp[j][i] ** 2 - dp[j][i - 1] ** 2)
                else:
                    dp[j][i] = dp[j][i - 1]
        return ans