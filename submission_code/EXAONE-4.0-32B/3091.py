from collections import Counter

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        mod = 10**9 + 7
        max_sum = 20000
        dp = [0] * (max_sum + 1)
        dp[0] = 1
        
        freq = Counter(nums)
        zeros_count = freq.get(0, 0)
        
        for x, cnt in freq.items():
            if x == 0:
                continue
            dp_old = dp.copy()
            for rem in range(x):
                j = rem
                while j <= max_sum:
                    if j >= x:
                        dp[j] = (dp[j] + dp[j - x]) % mod
                    if j >= (cnt + 1) * x:
                        dp[j] = (dp[j] - dp_old[j - (cnt + 1) * x]) % mod
                    j += x
        
        if zeros_count > 0:
            multiplier = zeros_count + 1
            for i in range(len(dp)):
                dp[i] = (dp[i] * multiplier) % mod
        
        total_ways = 0
        for s in range(l, r + 1):
            if s <= max_sum:
                total_ways = (total_ways + dp[s]) % mod
        
        return total_ways