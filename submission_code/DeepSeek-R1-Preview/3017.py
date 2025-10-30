class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count(x):
            if x < 0:
                return 0
            s = str(x)
            digits = list(map(int, s))
            n = len(digits)
            
            from functools import lru_cache
            
            @lru_cache(maxsize=None)
            def dp(pos, even, odd, leading_zero, tight, mod):
                if pos == n:
                    if even == odd and even + odd > 0 and mod % k == 0:
                        return 1
                    else:
                        return 0
                res = 0
                upper = digits[pos] if tight else 9
                for d in range(0, upper + 1):
                    new_tight = tight and (d == upper)
                    new_leading_zero = leading_zero and (d == 0)
                    new_mod = (mod * 10 + d) % k
                    
                    if new_leading_zero:
                        new_even = even
                        new_odd = odd
                    else:
                        if leading_zero:
                            if d % 2 == 0:
                                new_even = 1
                                new_odd = 0
                            else:
                                new_even = 0
                                new_odd = 1
                        else:
                            if d % 2 == 0:
                                new_even = even + 1
                                new_odd = odd
                            else:
                                new_even = even
                                new_odd = odd + 1
                    if new_even > 5 or new_odd > 5:
                        continue
                    res += dp(pos + 1, new_even, new_odd, new_leading_zero, new_tight, new_mod)
                return res
            
            return dp(0, 0, 0, True, True, 0)
        
        return count(high) - count(low - 1)