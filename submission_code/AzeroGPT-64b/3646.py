class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        dp = defaultdict(int)
        dp[nums[0]] = nums[0]
        for num in nums[1:]:
            dp[num] += num * (1 + dp[num-1] + dp[num] + dp[num+1])%mod
            dp[num]=dp[num]%mod
        return sum(dp.values())%mod