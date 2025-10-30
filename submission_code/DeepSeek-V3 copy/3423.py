class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        def max_sum_subsequence(arr):
            n = len(arr)
            if n == 0:
                return 0
            dp = [0] * n
            dp[0] = max(0, arr[0])
            if n > 1:
                dp[1] = max(dp[0], arr[1])
            for i in range(2, n):
                dp[i] = max(dp[i-1], dp[i-2] + max(0, arr[i]))
            return dp[-1]
        
        total = 0
        for pos, x in queries:
            nums[pos] = x
            current_max = max_sum_subsequence(nums)
            total = (total + current_max) % MOD
        return total