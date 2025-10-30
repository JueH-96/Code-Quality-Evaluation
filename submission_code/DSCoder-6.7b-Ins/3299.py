class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        nums.sort()
        dp = [0] * (10**9+1)
        for num in nums:
            if num == 1:
                dp[num] = max(dp[num], 1)
            else:
                dp[num] = max(dp[num], dp[num//2] + 1 if num % 2 == 0 else dp[num//2])
            for i in range(num*2, 10**9+1, num):
                dp[i] = max(dp[i], dp[num] + 1)
        return max(dp[:max(nums)+1])