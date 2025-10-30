class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[i][t] = length of the longest "good" subsequence ending at index i using at most t transitions
        dp = [[1]*(k+1) for _ in range(n)]
        
        ans = 1  # At least one element can always form a subsequence
        for i in range(n):
            for t in range(k+1):
                for j in range(i):
                    if nums[j] == nums[i]:
                        # Appending nums[i] to a subsequence ending in the same value doesn't use a transition
                        dp[i][t] = max(dp[i][t], dp[j][t] + 1)
                    else:
                        # Appending nums[i] to a subsequence ending in a different value uses a transition
                        if t > 0:
                            dp[i][t] = max(dp[i][t], dp[j][t-1] + 1)
                ans = max(ans, dp[i][t])
        
        return ans