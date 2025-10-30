from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        dp_prev = [0] * 51
        first_val = nums[0]
        for x in range(first_val + 1):
            dp_prev[x] = 1
        
        pre_sum = [0] * 51
        pre_sum[0] = dp_prev[0]
        for x in range(1, 51):
            pre_sum[x] = (pre_sum[x-1] + dp_prev[x]) % MOD
        
        for i in range(1, n):
            current_num = nums[i]
            prev_num = nums[i-1]
            delta = max(0, current_num - prev_num)
            dp_curr = [0] * 51
            
            for y in range(current_num + 1):
                if y < delta:
                    dp_curr[y] = 0
                else:
                    upper = y - delta
                    upper = min(upper, prev_num)
                    if upper < 0:
                        dp_curr[y] = 0
                    else:
                        dp_curr[y] = pre_sum[upper] % MOD
            
            dp_prev = dp_curr[:]
            pre_sum[0] = dp_prev[0]
            for x in range(1, 51):
                pre_sum[x] = (pre_sum[x-1] + dp_prev[x]) % MOD
        
        total = sum(dp_prev[:nums[-1] + 1]) % MOD
        return total