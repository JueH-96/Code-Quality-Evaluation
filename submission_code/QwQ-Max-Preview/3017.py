class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def f(x):
            if x == 0:
                return 0
            s = str(x)
            n = len(s)
            
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos, tight, even_odd_diff, mod, leading_zero):
                if pos == n:
                    if leading_zero:
                        return 0
                    return 1 if (even_odd_diff == 0 and mod == 0) else 0
                
                res = 0
                upper = int(s[pos]) if tight else 9
                
                for d in range(0, upper + 1):
                    new_tight = tight and (d == upper)
                    new_leading_zero = leading_zero and (d == 0)
                    
                    if new_leading_zero:
                        res += dp(pos + 1, new_tight, even_odd_diff, mod, new_leading_zero)
                    else:
                        if leading_zero and d != 0:
                            new_mod = d % k
                            delta = 1 if d % 2 == 0 else -1
                            new_even_odd = even_odd_diff + delta
                        else:
                            new_mod = (mod * 10 + d) % k
                            delta = 1 if d % 2 == 0 else -1
                            new_even_odd = even_odd_diff + delta
                        
                        res += dp(pos + 1, new_tight, new_even_odd, new_mod, new_leading_zero)
                return res
            
            return dp(0, True, 0, 0, True)
        
        return f(high) - f(low - 1)