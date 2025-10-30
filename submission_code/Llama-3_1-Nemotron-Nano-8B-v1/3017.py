class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_beautiful(n):
            if n < 0:
                return 0
            s = str(n)
            digits = list(map(int, s))
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(pos, even_count, mod, tight, leading_zero, num_digits):
                if pos == len(digits):
                    if leading_zero:
                        return 0
                    if num_digits % 2 != 0:
                        return 0
                    if even_count != num_digits // 2:
                        return 0
                    if mod != 0:
                        return 0
                    return 1
                res = 0
                max_d = digits[pos] if tight else 9
                for d in range(0, max_d + 1):
                    new_tight = tight and (d == max_d)
                    new_leading_zero = leading_zero and (d == 0)
                    if leading_zero:
                        if d == 0:
                            new_even = even_count
                            new_mod = mod
                            new_nd = num_digits
                        else:
                            new_even = 1 if d % 2 == 0 else 0
                            new_mod = d % k
                            new_nd = 1
                    else:
                        new_even = even_count + (1 if d % 2 == 0 else 0)
                        new_mod = (mod * 10 + d) % k
                        new_nd = num_digits + 1
                    res += dp(pos + 1, new_even, new_mod, new_tight, new_leading_zero, new_nd)
                return res
            return dp(0, 0, 0, True, True, 0)
        
        return count_beautiful(high) - count_beautiful(low - 1)