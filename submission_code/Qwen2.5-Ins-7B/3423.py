class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        dp = [0] * (n + 1)
        dp[0] = 0
        dp[1] = nums[0]
        
        for i in range(2, n + 1):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])
        
        def update_and_query(pos, val):
            prev = nums[pos]
            nums[pos] = val
            dp = [0] * (n + 1)
            dp[0] = 0
            dp[1] = nums[0]
            for i in range(2, n + 1):
                dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])
            return dp[n]
        
        result = 0
        for pos, val in queries:
            result = (result + update_and_query(pos, val)) % mod
        return result