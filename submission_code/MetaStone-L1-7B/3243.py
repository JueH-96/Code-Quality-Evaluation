class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        s_num = int(s)
        k = len(s)
        
        power = 10 ** k
        numerator = start - s_num + (power - 1)
        m_min = max(0, numerator // power)
        m_max = (finish - s_num) // power
        
        if m_min > m_max:
            return 0
        
        def count_numbers(n, limit):
            if n < 0:
                return 0
            s = str(n)
            digits = list(map(int, s))
            k_n = len(digits)
            
            from functools import lru_cache
            @lru_cache(maxsize=None)
            def dp(pos, tight, started):
                if pos == k_n:
                    return 1
                max_d = digits[pos] if tight else 9
                limit_d = min(max_d, limit)
                total = 0
                for d in range(0, limit_d + 1):
                    new_tight = tight and (d == max_d)
                    new_started = started or (d != 0)
                    total += dp(pos + 1, new_tight, new_started)
                return total
            
            return dp(0, True, False)
        
        count_m_max = count_numbers(m_max, limit)
        count_m_min_minus_1 = count_numbers(m_min - 1, limit)
        return count_m_max - count_m_min_minus_1