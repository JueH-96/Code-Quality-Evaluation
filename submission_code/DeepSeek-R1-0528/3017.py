class Solution:
    def dp_solve(self, L, bound, k):
        digits = list(map(int, bound))
        n = len(digits)
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def dp(pos, tight, diff, mod):
            if pos == n:
                return 1 if diff == 0 and mod == 0 else 0
            total = 0
            current_limit = digits[pos] if tight else 9
            for dgt in range(0, current_limit + 1):
                if pos == 0 and dgt == 0:
                    continue
                new_tight = tight and (dgt == current_limit)
                if dgt % 2 == 0:
                    new_diff = diff + 1
                else:
                    new_diff = diff - 1
                new_mod = (mod * 10 + dgt) % k
                total += dp(pos + 1, new_tight, new_diff, new_mod)
            return total
        return dp(0, True, 0, 0)
    
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def f(x):
            if x < 10:
                return 0
            s = str(x)
            d = len(s)
            res = 0
            for L in range(2, d + 1, 2):
                if L < d:
                    res += self.dp_solve(L, '9' * L, k)
                else:
                    res += self.dp_solve(L, s, k)
            return res
        
        return f(high) - f(low - 1)