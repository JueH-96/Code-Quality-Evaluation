import functools

class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_exactly_D(D):
            if D % 2 != 0:
                return 0
            @functools.lru_cache(None)
            def dp(pos, diff, rem):
                if pos == D:
                    return 1 if diff == 0 and rem == 0 else 0
                ans = 0
                if pos == 0:
                    for digit in range(1, 10):  # digits 1-9
                        new_diff = diff + (1 if digit % 2 == 0 else -1)
                        new_rem = (rem * 10 + digit) % k
                        ans += dp(pos + 1, new_diff, new_rem)
                else:
                    for digit in range(10):  # digits 0-9
                        new_diff = diff + (1 if digit % 2 == 0 else -1)
                        new_rem = (rem * 10 + digit) % k
                        ans += dp(pos + 1, new_diff, new_rem)
                return ans
            return dp(0, 0, 0)
        
        def count_up_to(X):
            if X < 0 or X == 0:
                return 0
            digits_str = str(X)
            digits = [int(c) for c in digits_str]
            L = len(digits)
            sum_smaller = 0
            for D in range(1, L):  # sum over digit lengths from 1 to L-1
                sum_smaller += count_exactly_D(D)
            # Now handle exactly L digits <= X
            @functools.lru_cache(None)
            def dp_tight(pos, diff, rem, tight_flag):
                if pos == L:
                    return 1 if diff == 0 and rem == 0 else 0
                ans = 0
                upper = digits[pos] if tight_flag == 1 else 9
                low_digit = 1 if pos == 0 else 0
                for digit in range(low_digit, upper + 1):
                    new_diff = diff + (1 if digit % 2 == 0 else -1)
                    new_rem = (rem * 10 + digit) % k
                    new_tight_flag = 1 if (tight_flag == 1 and digit == digits[pos]) else 0
                    ans += dp_tight(pos + 1, new_diff, new_rem, new_tight_flag)
                return ans
            count_exact_L = dp_tight(0, 0, 0, 1)
            return sum_smaller + count_exact_L
        
        return count_up_to(high) - count_up_to(low - 1)