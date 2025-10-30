from collections import Counter

MOD = 10**9 + 7

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        cnt = Counter(nums)
        unique_nums = sorted(cnt.keys())
        max_sum = r
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # empty subset
        
        for num in unique_nums:
            freq = cnt[num]
            new_dp = [0] * (max_sum + 1)
            for s in range(max_sum + 1):
                if dp[s]:
                    for k in range(freq + 1):
                        if s + k * num <= max_sum:
                            new_dp[s + k * num] = (new_dp[s + k * num] + dp[s]) % MOD
            dp = new_dp
        
        total = 0
        for s in range(l, r + 1):
            total = (total + dp[s]) % MOD
        return total