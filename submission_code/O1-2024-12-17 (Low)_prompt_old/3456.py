class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[i][c] = length of the longest "good" subsequence ending exactly at index i,
        #            using c "changes" so far (where a change is a transition between
        #            consecutive elements of different values in the subsequence).
        dp = [[0] * (k + 1) for _ in range(n)]
        
        answer = 1  # At least one element can always be taken.
        for i in range(n):
            # We can always start a new subsequence at i with 0 changes used, length = 1
            dp[i][0] = 1
            # Try to extend from all j < i
            for j in range(i):
                for c in range(k + 1):
                    if dp[j][c] == 0:
                        continue
                    if nums[j] == nums[i]:
                        # No new change if the values match
                        dp[i][c] = max(dp[i][c], dp[j][c] + 1)
                    else:
                        # We incur a new change if we switch values
                        if c < k:
                            dp[i][c + 1] = max(dp[i][c + 1], dp[j][c] + 1)
            
            # Update global answer
            answer = max(answer, max(dp[i]))
        
        return answer