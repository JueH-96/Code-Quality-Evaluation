class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_beautiful_integers(n: int) -> int:
            from functools import cache

            @cache
            def dp(pos: int, tight: bool, non_zero: bool, even_count: int, odd_count: int, remainder: int) -> int:
                if pos == len(n):
                    return int(non_zero and even_count == odd_count and remainder == 0)

                limit = int(n[pos]) if tight else 9
                total = 0

                for digit in range(limit + 1):
                    new_tight = tight and digit == limit
                    new_non_zero = non_zero or digit > 0
                    new_even_count = even_count + (digit % 2 == 0)
                    new_odd_count = odd_count + (digit % 2 != 0)
                    new_remainder = (remainder * 10 + digit) % k

                    total += dp(pos + 1, new_tight, new_non_zero, new_even_count, new_odd_count, new_remainder)

                return total

            return dp(0, True, False, 0, 0, 0)

        return count_beautiful_integers(str(high)) - count_beautiful_integers(str(low - 1))