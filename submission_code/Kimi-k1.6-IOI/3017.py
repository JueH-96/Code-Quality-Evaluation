from functools import lru_cache

class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def f(x):
            if x < 0:
                return 0
            s = str(x)
            n = len(s)
            total = 0
            
            # Count all even m < n
            for m in range(2, n, 2):
                total += count_all(m)
            
            # Count m = n if even
            if n % 2 == 0:
                total += count_up_to(s)
            
            return total
        
        def count_all(m):
            if m % 2 != 0:
                return 0
            @lru_cache(maxsize=None)
            def dp(pos, diff, mod):
                if pos == m:
                    return 1 if diff == 0 and mod == 0 else 0
                res = 0
                start = 0 if pos != 0 else 1
                for d in range(start, 10):
                    new_diff = diff + (1 if d % 2 == 0 else -1)
                    new_mod = (mod * 10 + d) % k
                    res += dp(pos + 1, new_diff, new_mod)
                return res
            return dp(0, 0, 0)
        
        def count_up_to(s):
            m = len(s)
            if m % 2 != 0:
                return 0
            @lru_cache(maxsize=None)
            def dp(pos, tight, diff, mod):
                if pos == m:
                    return 1 if diff == 0 and mod == 0 else 0
                limit = int(s[pos]) if tight else 9
                start = 0 if pos != 0 else 1
                total = 0
                for d in range(start, limit + 1):
                    new_tight = tight and (d == limit)
                    new_diff = diff + (1 if d % 2 == 0 else -1)
                    new_mod = (mod * 10 + d) % k
                    total += dp(pos + 1, new_tight, new_diff, new_mod)
                return total
            return dp(0, True, 0, 0)
        
        return f(high) - f(low - 1)