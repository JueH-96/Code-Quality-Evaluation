from collections import Counter

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        cnt = Counter(nums)
        unique_nums = sorted(cnt.keys())
        max_sum = r
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # empty subset
        
        for num in unique_nums:
            freq = cnt[num]
            for s in range(max_sum, -1, -1):
                if dp[s]:
                    for k in range(1, freq + 1):
                        if s + num * k <= max_sum:
                            dp[s + num * k] = (dp[s + num * k] + dp[s]) % MOD
                        else:
                            break
        
        total = 0
        for s in range(l, r + 1):
            total = (total + dp[s]) % MOD
        return total