MOD = 10**9 + 7

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        max_val = 100000
        dp_count = [0] * (max_val + 2)
        dp_sum = [0] * (max_val + 2)
        total_ans = 0
        
        for num in nums:
            cnt = 1
            s = num
            
            if num - 1 >= 0:
                cnt = (cnt + dp_count[num - 1]) % MOD
                s = (s + dp_sum[num - 1] + dp_count[num - 1] * num) % MOD
            if num + 1 <= max_val:
                cnt = (cnt + dp_count[num + 1]) % MOD
                s = (s + dp_sum[num + 1] + dp_count[num + 1] * num) % MOD
            
            total_ans = (total_ans + s) % MOD
            
            dp_count[num] = (dp_count[num] + cnt) % MOD
            dp_sum[num] = (dp_sum[num] + s) % MOD
        
        return total_ans