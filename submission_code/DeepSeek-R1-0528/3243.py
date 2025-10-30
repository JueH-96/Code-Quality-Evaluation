class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        L = len(s)
        suffix_val = int(s)
        if finish < suffix_val:
            return 0
        base = 10 ** L
        p_min = (start - suffix_val + base - 1) // base
        p_max = (finish - suffix_val) // base
        
        if p_min > p_max:
            return 0
        
        def count_le(n, limit):
            if n < 0:
                return 0
            digits = list(map(int, str(n)))
            from functools import lru_cache
            @lru_cache(maxsize=None)
            def dfs(pos, tight):
                if pos == len(digits):
                    return 1
                total = 0
                upper = digits[pos] if tight else 9
                max_d = min(upper, limit) if tight else limit
                for d in range(0, max_d + 1):
                    next_tight = tight and (d == upper)
                    total += dfs(pos + 1, next_tight)
                return total
            return dfs(0, True)
        
        return count_le(p_max, limit) - count_le(p_min - 1, limit)