class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        freq = [0] * (k + 1)
        dp = [0] + [-1] * (1 << k)
        for i in range(k):
            freq[i] = nums.count(i)
        for i in range(k):
            dp_new = dp[:]
            for j in range(1 << k):
                if dp[j] == -1:
                    continue
                for x in range(k):
                    if (j >> x) & 1:
                        continue
                    dp_new[j | (1 << x)] = max(dp_new[j | (1 << x)], dp[j] + freq[x])
            dp = dp_new
        return n - dp[-1]