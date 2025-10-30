class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        from collections import defaultdict
        dp = defaultdict(lambda: (0, 0))  # (count, sum)
        
        for x in nums:
            count_minus, sum_minus = dp[x - 1]
            count_plus, sum_plus = dp[x + 1]
            
            new_count = (count_minus + count_plus + 1) % MOD
            new_sum = (sum_minus + count_minus * x + sum_plus + count_plus * x + x) % MOD
            
            current_count, current_sum = dp[x]
            dp[x] = (
                (current_count + new_count) % MOD,
                (current_sum + new_sum) % MOD
            )
        
        total = 0
        for count, sum_val in dp.values():
            total = (total + sum_val) % MOD
        
        return total