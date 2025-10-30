class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[i][j] represents the maximum length of a good subsequence ending at index i with exactly j changes
        dp = [[0] * (k + 1) for _ in range(n)]
        # Initialize dp[i][0] to 1 for all i, since no changes are allowed
        for i in range(n):
            dp[i][0] = 1
        
        # To keep track of the maximum length for each number and each change count
        max_len = [{} for _ in range(k + 1)]
        for i in range(n):
            for j in range(k + 1):
                # If we have no changes, we can only take the current number if it's the same as the previous
                if j == 0:
                    dp[i][j] = 1
                else:
                    # We can either take the current number and not change, or take it and change
                    # For not changing, we look for the same number in the previous step
                    if nums[i] in max_len[j]:
                        dp[i][j] = max(dp[i][j], max_len[j][nums[i]] + 1)
                    # For changing, we look for any number in the previous step with j-1 changes
                    if j > 0:
                        for num in max_len[j-1]:
                            dp[i][j] = max(dp[i][j], max_len[j-1][num] + 1)
                # Update the max_len for the current number and change count
                if nums[i] not in max_len[j]:
                    max_len[j][nums[i]] = dp[i][j]
                else:
                    max_len[j][nums[i]] = max(max_len[j][nums[i]], dp[i][j])
        
        # The answer is the maximum value in the last row of dp
        result = 0
        for i in range(n):
            for j in range(k + 1):
                result = max(result, dp[i][j])
        return result