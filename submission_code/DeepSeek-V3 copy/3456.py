class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[i][j] represents the maximum length of a subsequence ending with nums[i] and having j changes
        dp = [[0] * (k + 1) for _ in range(n)]
        # Initialize dp[i][0] for all i
        for i in range(n):
            dp[i][0] = 1
        # Initialize the maximum length for each number and each change count
        max_len = [{} for _ in range(k + 1)]
        for i in range(n):
            for j in range(k + 1):
                # If we can take nums[i] without increasing the change count
                if nums[i] in max_len[j]:
                    dp[i][j] = max(dp[i][j], max_len[j][nums[i]] + 1)
                # If we can take nums[i] by increasing the change count
                if j > 0:
                    for num in max_len[j - 1]:
                        if num != nums[i]:
                            dp[i][j] = max(dp[i][j], max_len[j - 1][num] + 1)
                # Update the max_len[j] with the current dp[i][j]
                if nums[i] not in max_len[j] or dp[i][j] > max_len[j][nums[i]]:
                    max_len[j][nums[i]] = dp[i][j]
        # Find the maximum length among all dp[i][j]
        result = 0
        for i in range(n):
            for j in range(k + 1):
                result = max(result, dp[i][j])
        return result