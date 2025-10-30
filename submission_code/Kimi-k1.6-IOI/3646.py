class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        from collections import defaultdict
        count = defaultdict(int)
        sum_s = defaultdict(int)
        
        for y in nums:
            prev_minus_count = count[y - 1]
            prev_plus_count = count[y + 1]
            prev_minus_sum = sum_s[y - 1]
            prev_plus_sum = sum_s[y + 1]
            
            new_count = (prev_minus_count + prev_plus_count + 1) % MOD
            new_sum = (prev_minus_sum + prev_plus_sum + y * (prev_minus_count + prev_plus_count + 1)) % MOD
            
            count[y] = (count[y] + new_count) % MOD
            sum_s[y] = (sum_s[y] + new_sum) % MOD
        
        total = sum(sum_s.values()) % MOD
        return total