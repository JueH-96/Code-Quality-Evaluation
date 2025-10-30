class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(k + 1):
                dp[i][j] = max(dp[i][j], dp[i - 1][j])
                for l in range(1, i + 1):
                    count = 0
                    freq = {}
                    max_freq = 0
                    for m in range(i - l, i):
                        if nums[m] not in freq:
                            freq[nums[m]] = 0
                        freq[nums[m]] += 1
                        max_freq = max(max_freq, freq[nums[m]])
                    count = l - max_freq
                    if j >= count:
                        dp[i][j] = max(dp[i][j], dp[i-l][j-count] + l)
        
        return dp[n][k]