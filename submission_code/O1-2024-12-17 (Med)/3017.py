class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        """
        We use a digit-DP (dynamic programming) approach to count how many
        integers up to a certain value n satisfy both conditions:
          1. The count of even digits equals the count of odd digits.
          2. The integer is divisible by k.

        Then we compute:
            count_up_to(high) - count_up_to(low - 1)

        The DP state tracks:
            - pos: current position in the digit array (0-based, left to right).
            - leading: whether we have not yet placed any nonzero digit.
            - rem: current remainder (mod k).
            - diff: difference (#even_digits - #odd_digits).
            - tight: whether we are bound by the current prefix of n or free to choose [0..9].

        At the end of the digits, if we have formed at least one digit (leading == False),
        and rem == 0, and diff == 0, we count that number as beautiful.
        """

        from functools import lru_cache

        def get_digits(x: int) -> list:
            return list(map(int, str(x)))

        def count_up_to(x: int) -> int:
            # For non-positive x, no valid positive numbers exist
            if x <= 0:
                return 0

            digits = get_digits(x)

            @lru_cache(None)
            def dfs(pos: int, leading: bool, rem: int, diff: int, tight: bool) -> int:
                # If we've considered all digits:
                if pos == len(digits):
                    # Valid if we've formed a number and both conditions satisfy
                    return 1 if (not leading and rem == 0 and diff == 0) else 0

                limit = digits[pos] if tight else 9
                total = 0
                for dig in range(limit + 1):
                    n_leading = leading
                    n_rem = rem
                    n_diff = diff
                    n_tight = tight

                    # If still leading zeros and pick 0 => no changes
                    if n_leading and dig == 0:
                        pass
                    else:
                        # We place a nonzero digit or we have already placed digits
                        if n_leading:
                            n_leading = False
                        if dig % 2 == 0:
                            # Even digit
                            n_diff += 1
                        else:
                            # Odd digit
                            n_diff -= 1
                        n_rem = (n_rem * 10 + dig) % k

                    # If we choose a digit less than the limit, we can no longer be tight
                    if dig < limit:
                        n_tight = False

                    total += dfs(pos + 1, n_leading, n_rem, n_diff, n_tight)

                return total

            return dfs(0, True, 0, 0, True)

        # Count beautiful numbers in [low, high]
        return count_up_to(high) - count_up_to(low - 1)