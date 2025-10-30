class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]

        for i in range(1, n):
            dp[i] = nums[i]
            for j in range(i):
                score = dp[j] + nums[i]
                if nums[i] % 2 != nums[j] % 2:
                    score -= x
                dp[i] = max(dp[i], score)
        
        return max(dp)