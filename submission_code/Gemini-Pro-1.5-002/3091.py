class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        counts = Counter(nums)
        keys = sorted(counts.keys())
        n = len(keys)
        dp = [[0] * (r + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            num = keys[i - 1]
            count = counts[num]
            for j in range(r + 1):
                for k in range(count + 1):
                    if j - num * k >= 0:
                        dp[i][j] = (dp[i][j] + dp[i - 1][j - num * k]) % MOD
        ans = 0
        for i in range(l, r + 1):
            ans = (ans + dp[n][i]) % MOD
        return ans