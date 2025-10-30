from collections import Counter

MOD = 10**9 + 7

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        freq = Counter(nums)
        count_zero = freq.get(0, 0)
        if 0 in freq:
            del freq[0]
        
        sum_non_zero = sum(x * c for x, c in freq.items())
        if sum_non_zero < l:
            return 0
        
        max_sum = sum_non_zero
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # Empty subset
        
        for x in freq:
            c = freq[x]
            old_dp = dp.copy()
            for r_remainder in range(x):
                window_sum = 0
                for j in range(r_remainder, max_sum + 1, x):
                    if j >= x:
                        window_sum += old_dp[j - x]
                        window_sum %= MOD
                    if j >= (c + 1) * x:
                        window_sum -= old_dp[j - (c + 1) * x]
                        window_sum %= MOD
                    # Ensure window_sum is non-negative
                    if window_sum < 0:
                        window_sum += MOD
                    dp[j] = (dp[j] + window_sum) % MOD
        
        # Multiply by (count_zero + 1)
        if count_zero > 0:
            dp = [(count * (count_zero + 1)) % MOD for count in dp]
        
        lower = max(l, 0)
        upper = min(r, max_sum)
        if lower > upper:
            return 0
        total = sum(dp[lower : upper + 1]) % MOD
        return total