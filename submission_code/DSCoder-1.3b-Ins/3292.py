class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        changeIndices.append(n+1)
        changeIndices.insert(0, 0)
        nums = [0] + nums
        dp = [0] * (n + 2)
        for i in range(1, n + 1):
            if i < changeIndices[1]:
                dp[i] = dp[i - 1]
            else:
                dp[i] = min(dp[i - 1], dp[changeIndices[i // 2]] + 1)
        for i in range(1, m + 1):
            if dp[changeIndices[i]] != dp[changeIndices[i - 1]] + 1:
                return -1
        return dp[n] if dp[n] != float('inf') else -1