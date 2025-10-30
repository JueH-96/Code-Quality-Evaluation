from functools import lru_cache

class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count(n):
            s = list(map(int, str(n)))
            
            @lru_cache(maxsize=None)
            def dp(pos, even_minus_odd, rem, started, tight, digits_count):
                if pos == len(s):
                    if digits_count % 2 == 0 and digits_count != 0 and even_minus_odd == 0 and rem == 0:
                        return 1
                    else:
                        return 0
                res = 0
                max_digit = s[pos] if tight else 9
                for d in range(0, max_digit + 1):
                    new_tight = tight and (d == max_digit)
                    new_started = started or (d != 0)
                    
                    if not new_started:
                        # leading zero, so digits_count remains 0
                        new_digits_count = 0
                        new_even_minus_odd = even_minus_odd
                        new_rem = rem  # remains 0
                    else:
                        # compute new_digits_count
                        if started:
                            new_digits_count = digits_count + 1
                        else:
                            new_digits_count = 1
                        
                        # compute new_even_minus_odd
                        if d % 2 == 0:
                            delta = 1
                        else:
                            delta = -1
                        new_even_minus_odd = even_minus_odd + delta
                        
                        # compute new_rem
                        if started:
                            new_rem = (rem * 10 + d) % k
                        else:
                            new_rem = d % k
                    
                    res += dp(pos + 1, new_even_minus_odd, new_rem, new_started, new_tight, new_digits_count)
                return res
            
            return dp(0, 0, 0, False, True, 0)
        
        return count(high) - count(low - 1)