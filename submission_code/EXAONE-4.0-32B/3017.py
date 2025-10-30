class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        return self.count_up_to(high, k) - self.count_up_to(low - 1, k)
    
    def count_up_to(self, x: int, k: int) -> int:
        if x < 1:
            return 0
        s = str(x)
        n = len(s)
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def dp(pos, tight, started, mod, diff):
            if pos == n:
                if started and diff == 0 and mod % k == 0:
                    return 1
                return 0
            total = 0
            upper = int(s[pos]) if tight else 9
            for d in range(0, upper + 1):
                new_tight = tight and (d == upper)
                if not started and d == 0:
                    new_mod = (mod * 10 + d) % k
                    new_diff = diff
                    total += dp(pos + 1, new_tight, False, new_mod, new_diff)
                else:
                    new_started = True
                    new_mod = (mod * 10 + d) % k
                    if d % 2 == 0:
                        new_diff = diff + 1
                    else:
                        new_diff = diff - 1
                    total += dp(pos + 1, new_tight, new_started, new_mod, new_diff)
            return total
        return dp(0, True, False, 0, 0)