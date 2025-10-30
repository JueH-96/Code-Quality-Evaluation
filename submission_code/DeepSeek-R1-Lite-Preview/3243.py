import math
from functools import lru_cache

class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        len_s = len(s)
        s_num = int(s)
        
        # Calculate 10^len_s
        power = 10 ** len_s
        
        # If s_num > finish, no such k exists
        if s_num > finish:
            return 0
        
        # Calculate k_min: ceil((start - s_num) / power)
        numerator = start - s_num
        if numerator > 0:
            k_min = (numerator + power - 1) // power  # Ceiling division
        else:
            k_min = 0
        
        # Calculate k_max: floor((finish - s_num) / power)
        if finish < s_num:
            k_max = -1
        else:
            k_max = (finish - s_num) // power
        
        # If k_max < k_min, no such k exists
        if k_max < k_min:
            return 0
        
        # Helper function to count numbers <= N with each digit <= limit
        def count_up_to(N, limit):
            s_N = str(N)
            @lru_cache(maxsize=None)
            def dp(pos, tight):
                if pos == len(s_N):
                    return 1
                limit_digit = int(s_N[pos]) if tight else 9
                limit_digit = min(limit_digit, limit)
                total = 0
                for d in range(0, limit_digit + 1):
                    next_tight = tight and (d == limit_digit)
                    total += dp(pos + 1, next_tight)
                return total
            return dp(0, True)
        
        # Count numbers in [k_min, k_max]
        count_upper = count_up_to(k_max, limit)
        if k_min > 0:
            count_lower = count_up_to(k_min - 1, limit)
        else:
            count_lower = 0
        result = count_upper - count_lower
        return result