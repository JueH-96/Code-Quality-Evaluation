from typing import List

mod = 10**9 + 7

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        precomputed = [0] * (n + 1)
        dp = [0] * k
        dp[0] = 1
        precomputed[0] = 1
        total_sum = 1
        
        for n_val in range(1, n + 1):
            for j in range(k - 1, 0, -1):
                old_val = dp[j]
                new_val = dp[j] + dp[j - 1]
                if new_val >= mod:
                    new_val %= mod
                dp[j] = new_val
                total_sum = (total_sum + new_val - old_val) % mod
            precomputed[n_val] = total_sum
        
        nums.sort()
        result = 0
        for i in range(n):
            count_min = precomputed[n - i - 1]
            count_max = precomputed[i]
            result = (result + nums[i] * (count_min + count_max)) % mod
        
        return result