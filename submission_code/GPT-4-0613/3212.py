class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        dp, last = [0]*(len(nums)+1), {}
        dp[0] = 1
        for i in range(len(nums)):
            dp[i+1] = dp[i]*2
            if nums[i] in last:
                dp[i+1] -= dp[last[nums[i]]]
            dp[i+1] %= mod
            last[nums[i]] = i
        return dp[-1]