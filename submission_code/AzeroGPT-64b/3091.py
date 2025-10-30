class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        from bisect import bisect_left, bisect_right
        mod = 10**9 + 7
        freq = [0] * (max(nums) + 1)
        for x in nums:
            freq[x] += 1
        n = sum(freq) + 1
        dp = [[0] * n for _ in range(len(freq) + 1)]
        dp[0][0] = 1
        for i, f in enumerate(freq, 1):
            ith_sum = i
            for s1 in range(n):
                if s1 >= ith_sum:
                    dp[i][s1] = (dp[i-1][s1] + dp[i][s1-ith_sum]) % mod
                else:
                    dp[i][s1] = dp[i-1][s1]
                if f > 1:
                    dp[i][s1] += dp[i][s1-ith_sum] * (f-1)
                    dp[i][s1] %= mod
            for s1 in range(n-ith_sum):
                dp[i][s1+ith_sum] += dp[i][s1]
                dp[i][s1+ith_sum] %= mod
                
        return (bisect_right(dp[-1], r) - bisect_left(dp[-1], l)) % mod