class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        dp = [0, 0]  # dp[0] for subsequences ending with an odd sum, dp[1] for even sum
        last = [0, 0]  # last[0] for last odd sum, last[1] for last even sum
        for num in nums:
            if num % 2 == 0:
                dp[1] = last[0] + 1
                dp[0] = max(dp[0], last[1])
            else:
                dp[0] = last[1] + 1
                dp[1] = max(dp[1], last[0])
            last[num % 2] = dp[num % 2]
        return max(dp)