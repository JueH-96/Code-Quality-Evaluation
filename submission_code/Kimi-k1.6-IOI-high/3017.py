class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def query(L, x):
            min_L = 10 ** (L - 1)
            if x < min_L:
                return 0
            x_upper = min(x, 10**L - 1)
            s = str(x_upper)
            digits = tuple(int(c) for c in s)
            n = len(digits)
            
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos, tight, even_odd, mod):
                if pos == n:
                    return 1 if even_odd == 0 and mod == 0 else 0
                limit = digits[pos] if tight else 9
                total = 0
                for d in range(0, limit + 1):
                    if pos == 0 and d == 0:
                        continue
                    new_tight = tight and (d == limit)
                    new_even = even_odd + (1 if d % 2 == 0 else -1)
                    new_mod = (mod * 10 + d) % k
                    total += dp(pos + 1, new_tight, new_even, new_mod)
                return total
            
            return dp(0, True, 0, 0)
        
        res = 0
        max_digits = len(str(high))
        even_Ls = []
        for L in range(2, max_digits + 1):
            if L % 2 == 0:
                even_Ls.append(L)
        
        for L in even_Ls:
            min_L = 10 ** (L - 1)
            max_L = 10 ** L - 1
            lower = max(low, min_L)
            upper = min(high, max_L)
            if lower > upper:
                continue
            count_upper = query(L, upper)
            count_lower = query(L, lower - 1) if (lower - 1) >= 0 else 0
            res += count_upper - count_lower
        
        return res