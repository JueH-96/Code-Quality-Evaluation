from collections import defaultdict

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        dp = defaultdict(lambda: [0, 0])  # [count, sum]
        
        for num in nums:
            new_dp = defaultdict(lambda: [0, 0])
            # Include the current number as a new subsequence
            new_dp[num][0] += 1
            new_dp[num][1] += num
            # Extend existing subsequences
            for key in dp:
                if abs(key - num) == 1:
                    new_dp[num][0] += dp[key][0]
                    new_dp[num][1] += dp[key][1] + num * dp[key][0]
            # Merge with existing dp
            for key in new_dp:
                dp[key][0] = (dp[key][0] + new_dp[key][0]) % MOD
                dp[key][1] = (dp[key][1] + new_dp[key][1]) % MOD
        
        total = 0
        for key in dp:
            total = (total + dp[key][1]) % MOD
        return total