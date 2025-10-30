class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        n = len(nums)
        
        dp = [0] * n
        dp[0] = max(0, nums[0])
        if n > 1:
            dp[1] = max(dp[0], nums[1])
            for i in range(2, n):
                dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        res = 0
        for pos_i, x_i in queries:
            nums[pos_i] = x_i
            old_dp = dp[pos_i]
            if pos_i == 0:
                dp[0] = max(0, nums[0])
                i = 1
            else:
                dp[pos_i] = max(dp[pos_i-1], (dp[pos_i-2] if pos_i >=2 else 0) + nums[pos_i])
                i = pos_i +1
            while i < n:
                new_dp = max(dp[i-1], dp[i-2] + nums[i] if i >= 2 else nums[i])
                if dp[i] == new_dp:
                    break
                dp[i] = new_dp
                i +=1
            res = (res + dp[-1]) % mod
        return res