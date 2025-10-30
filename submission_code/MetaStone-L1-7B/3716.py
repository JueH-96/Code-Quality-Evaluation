class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        MAX_DIFF = 300
        dp = [[0] * MAX_DIFF for _ in range(n)]
        max_len = [[0] * MAX_DIFF for _ in range(n)]
        max_len_so_far = 0
        
        for i in range(n):
            # Update dp[i] based on previous elements
            for j in range(i):
                d = abs(nums[i] - nums[j])
                if d > MAX_DIFF:
                    d = MAX_DIFF  # Cap the difference to avoid index out of range
                if max_len[j][d] + 1 > dp[i][d]:
                    dp[i][d] = max_len[j][d] + 1
            # Compute max_len[i] as the maximum for each possible difference
            current_max = 0
            for d in range(MAX_DIFF):
                if dp[i][d] > current_max:
                    current_max = dp[i][d]
                max_len[i][d] = current_max
            # Update the global maximum
            for d in range(MAX_DIFF):
                if dp[i][d] > max_len_so_far:
                    max_len_so_far = dp[i][d]
        
        return max_len_so_far