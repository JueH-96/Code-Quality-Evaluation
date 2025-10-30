class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        total_sum_of_answers = 0
        current_nums = list(nums) # work with a copy to avoid modifying original nums for subsequent queries
        for query in queries:
            pos_i, x_i = query
            current_nums[pos_i] = x_i
            n = len(current_nums)
            if n == 0:
                current_answer = 0
            else:
                dp = [0] * n
                dp[0] = max(current_nums[0], 0)
                if n >= 2:
                    dp[1] = max(current_nums[1], dp[0])
                    for i in range(2, n):
                        dp[i] = max(dp[i-2] + current_nums[i], dp[i-1])
                current_answer = dp[n-1]
            total_sum_of_answers = (total_sum_of_answers + current_answer) % MOD
        return total_sum_of_answers