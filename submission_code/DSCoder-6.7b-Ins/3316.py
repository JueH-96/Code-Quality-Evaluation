class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        dp = [[0]*n for _ in range(k+1)]
        for i in range(n):
            dp[1][i] = nums[i]
        for i in range(2, k+1):
            for j in range(n-i+1):
                dp[i][j] = min(dp[i-1][j], dp[i-1][j+1])
        return sum(dp[k]) % (10**9 + 7)