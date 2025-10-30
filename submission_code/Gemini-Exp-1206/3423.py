class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        total_sum = 0

        def calculate_max_sum(arr):
            n = len(arr)
            if n == 0:
                return 0
            if n == 1:
                return max(0, arr[0])

            dp = [0] * n
            dp[0] = max(0, arr[0])
            dp[1] = max(dp[0], max(0, arr[1]))

            for i in range(2, n):
                dp[i] = max(dp[i - 1], dp[i - 2] + max(0, arr[i]))

            return dp[n - 1]

        for pos, x in queries:
            nums[pos] = x
            total_sum = (total_sum + calculate_max_sum(nums)) % MOD

        return total_sum