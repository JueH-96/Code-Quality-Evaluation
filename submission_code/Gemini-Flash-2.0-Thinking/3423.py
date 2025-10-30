class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        total_sum = 0
        mod = 10**9 + 7

        current_nums = list(nums)

        for pos, x in queries:
            current_nums[pos] = x

            n = len(current_nums)
            if n == 0:
                max_sum = 0
            elif n == 1:
                max_sum = max(0, current_nums[0])
            else:
                dp = [[0] * 2 for _ in range(n)]
                dp[0][0] = 0
                dp[0][1] = current_nums[0]

                for i in range(1, n):
                    dp[i][0] = max(dp[i - 1][0], dp[i - 1][1])
                    dp[i][1] = dp[i - 1][0] + current_nums[i]

                max_sum = max(dp[n - 1][0], dp[n - 1][1])

            total_sum = (total_sum + max_sum) % mod

        return total_sum