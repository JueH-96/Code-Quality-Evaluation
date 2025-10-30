class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[i][t] = the length of the longest good subsequence ending exactly at index i
        #            using at most t transitions so far.
        dp = [[1]*(k+1) for _ in range(n)]
        
        ans = 1  # There's at least one element, so answer is at least 1
        
        for i in range(n):
            for j in range(i):
                if nums[i] == nums[j]:
                    # No new transition if the values are the same
                    for t in range(k+1):
                        if dp[j][t] + 1 > dp[i][t]:
                            dp[i][t] = dp[j][t] + 1
                            ans = max(ans, dp[i][t])
                else:
                    # Transition if values differ (need t > 0 to allow one more transition)
                    for t in range(1, k+1):
                        if dp[j][t-1] + 1 > dp[i][t]:
                            dp[i][t] = dp[j][t-1] + 1
                            ans = max(ans, dp[i][t])
        
        return ans