class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        suffix_num = int(s)
        len_s = len(s)
        pow_10 = 10 ** len_s
        
        # Calculate m_min and m_max
        numerator_min = start - suffix_num
        if numerator_min <= 0:
            m_min = 0
        else:
            m_min = (numerator_min + pow_10 - 1) // pow_10
        
        numerator_max = finish - suffix_num
        if numerator_max < 0:
            m_max = -1
        else:
            m_max = numerator_max // pow_10
        
        if m_min > m_max:
            return 0
        
        def count_valid(a, b, limit):
            if a > b:
                return 0
            return count(b, limit) - count(a - 1, limit)
        
        def count(n, limit):
            if n < 0:
                return 0
            s_num = str(n)
            len_n = len(s_num)
            total = 0
            for k in range(1, len_n + 1):
                if k < len_n:
                    if k == 1:
                        upper = min(n, limit)
                        total += upper + 1 if upper >= 0 else 0
                    else:
                        if limit < 1:
                            continue
                        first = limit
                        others = (limit + 1) ** (k - 1)
                        total += first * others
                else:
                    if k == 1:
                        upper = min(n, limit)
                        total += upper + 1 if upper >= 0 else 0
                    else:
                        s_digits = list(map(int, s_num))
                        from functools import lru_cache
                        
                        @lru_cache(maxsize=None)
                        def dp(pos, tight):
                            if pos == k:
                                return 1
                            res = 0
                            upper = s_digits[pos] if tight else 9
                            for d in range(0, upper + 1):
                                if pos == 0:
                                    if d == 0 or d > limit:
                                        continue
                                else:
                                    if d > limit:
                                        continue
                                new_tight = tight and (d == upper)
                                res += dp(pos + 1, new_tight)
                            return res
                        
                        total += dp(0, True)
            return total
        
        return count_valid(m_min, m_max, limit)