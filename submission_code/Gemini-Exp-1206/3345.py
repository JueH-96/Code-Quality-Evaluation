class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = 10**9 + 7
        ans = 0
        for i in range(1, 1 << n):
            subsequence = []
            for j in range(n):
                if (i >> j) & 1:
                    subsequence.append(nums[j])
            
            m = len(subsequence)
            dp = [0] * (k + 1)
            dp[0] = 1
            count = 0
            for j in range(m):
                for s in range(k, subsequence[j] - 1, -1):
                    dp[s] = (dp[s] + dp[s - subsequence[j]]) % mod
            
            ans = (ans + dp[k]) % mod

        return ans