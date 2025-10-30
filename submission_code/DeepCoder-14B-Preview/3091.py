from collections import Counter

MOD = 10**9 + 7

class Solution:
    def countSubMultisets(self, nums: list, l: int, r: int) -> int:
        freq = Counter(nums)
        total_sum = sum(nums)
        max_sum = min(r, total_sum)
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # Base case: empty subset
        
        for x in freq:
            count = freq[x]
            # Create a temporary array to store the new state to avoid overwriting issues
            temp_dp = dp.copy()
            for s in range(max_sum, x - 1, -1):
                for k in range(1, count + 1):
                    if s >= x * k:
                        dp[s] = (dp[s] + temp_dp[s - x * k]) % MOD
            # Update the dp array with the new values
            dp = [ (dp[i] + temp_dp[i]) % MOD for i in range(max_sum + 1) ]
        
        result = sum(dp[l:r+1]) % MOD
        return result