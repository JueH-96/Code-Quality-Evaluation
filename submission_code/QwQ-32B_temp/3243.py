from functools import lru_cache

class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        L = len(s)
        s_num = int(s)
        if s_num > finish:
            return 0
        pow_L = 10 ** L
        
        # Calculate lower_N and upper_N
        numerator = start - s_num
        lower_N = (numerator + pow_L - 1) // pow_L
        lower_N = max(0, lower_N)
        upper_N = (finish - s_num) // pow_L
        
        if upper_N < lower_N:
            return 0
        
        def count(n, limit_digit):
            if n < 0:
                return 0
            s_n = str(n)
            
            @lru_cache(maxsize=None)
            def dfs(pos, tight, started):
                if pos == len(s_n):
                    return 1
                res = 0
                current_max = int(s_n[pos]) if tight else 9
                max_d = min(current_max, limit_digit)
                for d in range(0, max_d + 1):
                    new_tight = tight and (d == current_max)
                    new_started = started or (d != 0)
                    res += dfs(pos + 1, new_tight, new_started)
                return res
            return dfs(0, True, False)
        
        return count(upper_N, limit) - count(lower_N - 1, limit)